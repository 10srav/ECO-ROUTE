"""
EcoRoute SDN Controller - Main Ryu Application

Energy-Aware Dynamic Traffic Engineering Controller for SDN Data Center Networks.

Features:
- EWMA-based traffic prediction for proactive link sleep/wake
- Enhanced greedy routing with energy-aware path selection
- Make-before-break link transitions
- QoS-aware routing with utilization constraints
- Real-time statistics and metric export

Usage:
    ryu-manager ecoroute_controller.py --observe-links

Author: EcoRoute Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from threading import Thread
from typing import Dict, List, Optional, Set, Tuple

import yaml
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import (
    CONFIG_DISPATCHER,
    DEAD_DISPATCHER,
    MAIN_DISPATCHER,
    set_ev_cls,
)
from ryu.lib import hub
from ryu.lib.packet import arp, ethernet, icmp, ipv4, packet, tcp, udp
from ryu.ofproto import ofproto_v1_3
from ryu.topology import event as topo_event
from ryu.topology.api import get_all_link, get_all_switch

import structlog

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from controller.energy_model import EnergyModel, PortState
from controller.energy_router import EnergyAwareRouter, PathScore
from controller.ewma_predictor import AdaptiveEWMAPredictor
from controller.sleep_manager import SleepManager
from controller.stats_collector import FlowStats, PortStats, StatsCollector

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("ecoroute")


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    try:
        # Try relative to current working directory
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)

        # Try relative to script directory
        script_dir = Path(__file__).parent.parent
        full_path = script_dir / config_path
        if full_path.exists():
            with open(full_path, 'r') as f:
                return yaml.safe_load(f)

        logger.warning("config_not_found", path=config_path)
        return {}

    except Exception as e:
        logger.error("config_load_failed", path=config_path, error=str(e))
        return {}


class EcoRouteController(app_manager.RyuApp):
    """
    EcoRoute Energy-Aware SDN Controller

    Main Ryu application that coordinates all EcoRoute modules:
    - Traffic prediction (EWMA)
    - Energy-aware routing
    - Link sleep/wake management
    - Statistics collection
    """

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(EcoRouteController, self).__init__(*args, **kwargs)

        # Load configuration
        self.config = load_config()

        # Extract config values with defaults
        ewma_config = self.config.get("ewma", {})
        energy_config = self.config.get("energy", {})
        routing_config = self.config.get("routing", {})
        controller_config = self.config.get("controller", {})
        metrics_config = self.config.get("metrics", {})
        power_config = energy_config.get("power_model", {})

        # Initialize components
        self.predictor = AdaptiveEWMAPredictor(
            base_alpha=ewma_config.get("alpha", 0.3),
            min_alpha=ewma_config.get("adaptive_alpha", {}).get("min_alpha", 0.1),
            max_alpha=ewma_config.get("adaptive_alpha", {}).get("max_alpha", 0.7),
            prediction_steps=ewma_config.get("prediction_steps", 3),
            history_size=ewma_config.get("history_size", 100)
        )

        self.energy_model = EnergyModel(
            switch_base_power=power_config.get("switch_base_power", 50.0),
            port_power=power_config.get("port_power", 5.0),
            sleep_power=power_config.get("sleep_power", 0.5),
            wake_latency_ms=energy_config.get("wake_latency_ms", 100.0)
        )

        self.router = EnergyAwareRouter(
            energy_model=self.energy_model,
            predictor=self.predictor,
            k_paths=routing_config.get("k_paths", 3),
            energy_weight=routing_config.get("scoring", {}).get("energy_weight", 0.5),
            load_weight=routing_config.get("scoring", {}).get("load_weight", 0.3),
            hop_weight=routing_config.get("scoring", {}).get("hop_weight", 0.2),
            max_utilization=energy_config.get("qos", {}).get("max_utilization", 80.0)
        )

        self.sleep_manager = SleepManager(
            energy_model=self.energy_model,
            router=self.router,
            predictor=self.predictor,
            sleep_threshold=energy_config.get("sleep_threshold", 20.0),
            wake_threshold=energy_config.get("wake_threshold", 60.0),
            min_sleep_duration=energy_config.get("min_sleep_duration", 30.0),
            wake_latency_ms=energy_config.get("wake_latency_ms", 100.0),
            max_packet_loss=energy_config.get("qos", {}).get("max_packet_loss", 0.1)
        )

        self.stats_collector = StatsCollector(
            predictor=self.predictor,
            polling_interval=controller_config.get("stats_polling_interval", 5.0),
            export_path=metrics_config.get("export_path", "logs/metrics.csv"),
            export_interval=metrics_config.get("export_interval", 10.0)
        )

        # Set callbacks
        self.sleep_manager.set_flow_mod_callback(self._async_install_path)
        self.sleep_manager.set_port_mod_callback(self._async_port_mod)
        self.stats_collector.set_energy_callback(self.energy_model.get_stats)

        # Datapath tracking
        self.datapaths: Dict[int, any] = {}

        # MAC to port mapping per switch
        self.mac_to_port: Dict[int, Dict[str, int]] = {}

        # Host discovery: IP -> (dpid, port, MAC)
        self.hosts: Dict[str, Tuple[int, int, str]] = {}

        # ARP table: IP -> MAC
        self.arp_table: Dict[str, str] = {}

        # Flow ID counter
        self._flow_id_counter = 0

        # Polling interval
        self.stats_interval = controller_config.get("stats_polling_interval", 5)

        # Optimization interval
        self.optimization_interval = 10  # seconds

        # Running flag
        self._running = True

        # Start background threads
        self.stats_thread = hub.spawn(self._stats_polling_loop)
        self.optimization_thread = hub.spawn(self._optimization_loop)

        logger.info(
            "ecoroute_controller_initialized",
            stats_interval=self.stats_interval,
            optimization_interval=self.optimization_interval
        )

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Handle switch connection and install table-miss flow."""
        datapath = ev.msg.datapath
        dpid = datapath.id
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        self.datapaths[dpid] = datapath
        self.mac_to_port.setdefault(dpid, {})

        logger.info("switch_connected", dpid=dpid)

        # Install table-miss flow entry (send to controller)
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(
            ofproto.OFPP_CONTROLLER,
            ofproto.OFPCML_NO_BUFFER
        )]
        self._add_flow(datapath, 0, match, actions)

        # Get port descriptions for energy model
        self._request_port_desc(datapath)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, ev):
        """Handle switch state changes."""
        datapath = ev.datapath
        dpid = datapath.id

        if ev.state == MAIN_DISPATCHER:
            if dpid not in self.datapaths:
                self.datapaths[dpid] = datapath
                logger.info("switch_registered", dpid=dpid)

        elif ev.state == DEAD_DISPATCHER:
            if dpid in self.datapaths:
                del self.datapaths[dpid]
                self.energy_model.unregister_switch(dpid)
                logger.info("switch_disconnected", dpid=dpid)

    @set_ev_cls(ofp_event.EventOFPPortDescStatsReply, MAIN_DISPATCHER)
    def port_desc_stats_reply_handler(self, ev):
        """Handle port description reply for energy model initialization."""
        dpid = ev.msg.datapath.id
        ports = []

        for port in ev.msg.body:
            # Skip reserved ports
            if port.port_no < 65000:
                ports.append(port.port_no)

                # Set link capacity based on port config
                # Default to 1 Gbps, can be configured per port
                capacity = 1000.0  # Mbps
                self.stats_collector.set_link_capacity(dpid, port.port_no, capacity)

        # Register switch in energy model
        self.energy_model.register_switch(dpid, ports)

        logger.info(
            "switch_ports_discovered",
            dpid=dpid,
            ports=ports
        )

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """Handle packet-in events for unknown destinations."""
        msg = ev.msg
        datapath = msg.datapath
        dpid = datapath.id
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        if eth.ethertype == 0x88cc:  # LLDP
            return

        src_mac = eth.src
        dst_mac = eth.dst

        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src_mac] = in_port

        # Handle ARP packets for host discovery
        arp_pkt = pkt.get_protocol(arp.arp)
        if arp_pkt:
            self._handle_arp(datapath, in_port, eth, arp_pkt, msg.data)
            return

        # Handle IPv4 packets
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if ip_pkt:
            self._handle_ipv4(datapath, in_port, eth, ip_pkt, pkt, msg.data)
            return

        # Default: flood
        out_port = ofproto.OFPP_FLOOD
        actions = [parser.OFPActionOutput(out_port)]

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data
        )
        datapath.send_msg(out)

    def _handle_arp(self, datapath, in_port, eth, arp_pkt, data):
        """Handle ARP packets for host discovery."""
        dpid = datapath.id
        src_ip = arp_pkt.src_ip
        src_mac = arp_pkt.src_mac

        # Register host
        self.hosts[src_ip] = (dpid, in_port, src_mac)
        self.arp_table[src_ip] = src_mac
        self.router.add_host(src_ip, dpid, in_port)

        logger.debug(
            "host_discovered",
            ip=src_ip,
            mac=src_mac,
            dpid=dpid,
            port=in_port
        )

        if arp_pkt.opcode == arp.ARP_REQUEST:
            dst_ip = arp_pkt.dst_ip

            # Check if we know the destination
            if dst_ip in self.arp_table:
                # Reply on behalf of destination
                self._send_arp_reply(
                    datapath, in_port, eth,
                    dst_ip, self.arp_table[dst_ip],
                    src_ip, src_mac
                )
            else:
                # Flood ARP request
                self._flood_packet(datapath, in_port, data)

    def _send_arp_reply(
        self,
        datapath,
        out_port,
        eth,
        src_ip,
        src_mac,
        dst_ip,
        dst_mac
    ):
        """Send ARP reply."""
        parser = datapath.ofproto_parser

        arp_reply = packet.Packet()
        arp_reply.add_protocol(
            ethernet.ethernet(
                dst=dst_mac,
                src=src_mac,
                ethertype=0x0806
            )
        )
        arp_reply.add_protocol(
            arp.arp(
                opcode=arp.ARP_REPLY,
                src_mac=src_mac,
                src_ip=src_ip,
                dst_mac=dst_mac,
                dst_ip=dst_ip
            )
        )
        arp_reply.serialize()

        actions = [parser.OFPActionOutput(out_port)]
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=datapath.ofproto.OFP_NO_BUFFER,
            in_port=datapath.ofproto.OFPP_CONTROLLER,
            actions=actions,
            data=arp_reply.data
        )
        datapath.send_msg(out)

    def _handle_ipv4(self, datapath, in_port, eth, ip_pkt, pkt, data):
        """Handle IPv4 packets with energy-aware routing."""
        dpid = datapath.id
        src_ip = ip_pkt.src
        dst_ip = ip_pkt.dst

        # Register source host if not known
        if src_ip not in self.hosts:
            self.hosts[src_ip] = (dpid, in_port, eth.src)
            self.router.add_host(src_ip, dpid, in_port)

        # Check if destination is known
        if dst_ip not in self.hosts:
            # Destination unknown - flood
            self._flood_packet(datapath, in_port, data)
            return

        dst_dpid, dst_port, dst_mac = self.hosts[dst_ip]

        # Same switch - direct forward
        if dpid == dst_dpid:
            self._install_direct_flow(datapath, eth, ip_pkt, dst_port)
            self._send_packet(datapath, dst_port, data)
            return

        # Find energy-aware path
        path_score = self.router.find_best_path(dpid, dst_dpid)

        if not path_score:
            logger.warning(
                "no_path_found",
                src_dpid=dpid,
                dst_dpid=dst_dpid
            )
            self._flood_packet(datapath, in_port, data)
            return

        # Generate flow ID
        self._flow_id_counter += 1
        flow_id = f"flow_{self._flow_id_counter}"

        # Install flow rules along path
        self._install_path_flows(
            flow_id,
            path_score,
            eth.src,
            eth.dst,
            src_ip,
            dst_ip,
            dst_port
        )

        # Register flow in router
        self.router.install_flow(
            flow_id=flow_id,
            src_ip=src_ip,
            dst_ip=dst_ip,
            path_score=path_score,
            bandwidth=0.0  # Unknown initially
        )

        # Send first packet
        first_link = path_score.links[0] if path_score.links else None
        if first_link:
            out_port = first_link[1]  # src_port of first link
            self._send_packet(datapath, out_port, data)

    def _install_path_flows(
        self,
        flow_id: str,
        path_score: PathScore,
        src_mac: str,
        dst_mac: str,
        src_ip: str,
        dst_ip: str,
        final_port: int
    ):
        """Install flow rules along the computed path."""
        path = path_score.path
        links = path_score.links

        for i, dpid in enumerate(path):
            if dpid not in self.datapaths:
                continue

            datapath = self.datapaths[dpid]
            parser = datapath.ofproto_parser

            # Determine output port
            if i < len(links):
                out_port = links[i][1]  # src_port of link
            else:
                out_port = final_port

            # Create match for this flow
            match = parser.OFPMatch(
                eth_type=0x0800,
                ipv4_src=src_ip,
                ipv4_dst=dst_ip
            )

            actions = [parser.OFPActionOutput(out_port)]

            # Install with medium priority
            self._add_flow(datapath, 100, match, actions, idle_timeout=300)

            logger.debug(
                "flow_installed",
                flow_id=flow_id,
                dpid=dpid,
                out_port=out_port,
                src_ip=src_ip,
                dst_ip=dst_ip
            )

    def _install_direct_flow(self, datapath, eth, ip_pkt, out_port):
        """Install direct flow for same-switch forwarding."""
        parser = datapath.ofproto_parser

        match = parser.OFPMatch(
            eth_type=0x0800,
            ipv4_src=ip_pkt.src,
            ipv4_dst=ip_pkt.dst
        )

        actions = [parser.OFPActionOutput(out_port)]
        self._add_flow(datapath, 100, match, actions, idle_timeout=300)

    def _add_flow(
        self,
        datapath,
        priority: int,
        match,
        actions: List,
        idle_timeout: int = 0,
        hard_timeout: int = 0
    ):
        """Add a flow entry to a switch."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(
            ofproto.OFPIT_APPLY_ACTIONS,
            actions
        )]

        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=idle_timeout,
            hard_timeout=hard_timeout
        )
        datapath.send_msg(mod)

    def _delete_flow(self, datapath, match):
        """Delete a flow entry from a switch."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        mod = parser.OFPFlowMod(
            datapath=datapath,
            command=ofproto.OFPFC_DELETE,
            out_port=ofproto.OFPP_ANY,
            out_group=ofproto.OFPG_ANY,
            match=match
        )
        datapath.send_msg(mod)

    def _flood_packet(self, datapath, in_port, data):
        """Flood a packet to all ports except input."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=ofproto.OFP_NO_BUFFER,
            in_port=in_port,
            actions=actions,
            data=data
        )
        datapath.send_msg(out)

    def _send_packet(self, datapath, port, data):
        """Send a packet out a specific port."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        actions = [parser.OFPActionOutput(port)]
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=ofproto.OFP_NO_BUFFER,
            in_port=ofproto.OFPP_CONTROLLER,
            actions=actions,
            data=data
        )
        datapath.send_msg(out)

    def _request_port_desc(self, datapath):
        """Request port descriptions from a switch."""
        parser = datapath.ofproto_parser
        req = parser.OFPPortDescStatsRequest(datapath, 0)
        datapath.send_msg(req)

    def _request_port_stats(self, datapath):
        """Request port statistics from a switch."""
        parser = datapath.ofproto_parser
        req = parser.OFPPortStatsRequest(
            datapath, 0, datapath.ofproto.OFPP_ANY
        )
        datapath.send_msg(req)

    def _request_flow_stats(self, datapath):
        """Request flow statistics from a switch."""
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply_handler(self, ev):
        """Handle port statistics reply."""
        dpid = ev.msg.datapath.id
        current_time = time.time()

        port_stats = []
        for stat in ev.msg.body:
            port_stats.append(PortStats(
                dpid=dpid,
                port_no=stat.port_no,
                rx_packets=stat.rx_packets,
                tx_packets=stat.tx_packets,
                rx_bytes=stat.rx_bytes,
                tx_bytes=stat.tx_bytes,
                rx_dropped=stat.rx_dropped,
                tx_dropped=stat.tx_dropped,
                rx_errors=stat.rx_errors,
                tx_errors=stat.tx_errors,
                timestamp=current_time
            ))

        # Process stats
        metrics = self.stats_collector.process_port_stats(dpid, port_stats)

        # Update router with utilization
        for port_no, link_metrics in metrics.items():
            # Find the destination of this port
            for (src, dst), (src_port, _, _) in self.router._link_info.items():
                if src == dpid and src_port == port_no:
                    self.router.update_link_utilization(
                        src, dst,
                        link_metrics.utilization_percent
                    )
                    break

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        """Handle flow statistics reply."""
        dpid = ev.msg.datapath.id
        current_time = time.time()

        flow_stats = []
        for stat in ev.msg.body:
            flow_stats.append(FlowStats(
                dpid=dpid,
                table_id=stat.table_id,
                match=dict(stat.match),
                priority=stat.priority,
                byte_count=stat.byte_count,
                packet_count=stat.packet_count,
                duration_sec=stat.duration_sec,
                duration_nsec=stat.duration_nsec,
                timestamp=current_time
            ))

        self.stats_collector.process_flow_stats(dpid, flow_stats)

    @set_ev_cls(topo_event.EventLinkAdd, MAIN_DISPATCHER)
    def link_add_handler(self, ev):
        """Handle new link discovery."""
        src = ev.link.src
        dst = ev.link.dst

        # Add link to router
        self.router.add_link(
            src_dpid=src.dpid,
            src_port=src.port_no,
            dst_dpid=dst.dpid,
            dst_port=dst.port_no,
            capacity=1000.0  # Default 1 Gbps
        )

        logger.info(
            "link_discovered",
            src_dpid=src.dpid,
            src_port=src.port_no,
            dst_dpid=dst.dpid,
            dst_port=dst.port_no
        )

    @set_ev_cls(topo_event.EventLinkDelete, MAIN_DISPATCHER)
    def link_delete_handler(self, ev):
        """Handle link removal."""
        src = ev.link.src
        dst = ev.link.dst

        self.router.remove_link(src.dpid, dst.dpid)

        logger.info(
            "link_removed",
            src_dpid=src.dpid,
            dst_dpid=dst.dpid
        )

    def _stats_polling_loop(self):
        """Background loop for statistics polling."""
        while self._running:
            for dpid, datapath in list(self.datapaths.items()):
                try:
                    self._request_port_stats(datapath)
                    self._request_flow_stats(datapath)
                except Exception as e:
                    logger.error(
                        "stats_request_failed",
                        dpid=dpid,
                        error=str(e)
                    )

            # Export metrics
            self.stats_collector.export_metrics()

            hub.sleep(self.stats_interval)

    def _optimization_loop(self):
        """Background loop for sleep/wake optimization."""
        # Wait for topology discovery
        hub.sleep(30)

        while self._running:
            try:
                # Run optimization cycle using hub for async
                self._run_optimization()
            except Exception as e:
                logger.error(
                    "optimization_failed",
                    error=str(e)
                )

            hub.sleep(self.optimization_interval)

    def _run_optimization(self):
        """Run sleep/wake optimization (synchronous wrapper)."""
        # Get candidates
        sleep_candidates = self.sleep_manager.get_sleep_candidates()
        wake_candidates = self.sleep_manager.get_wake_candidates()

        logger.debug(
            "optimization_cycle",
            sleep_candidates=len(sleep_candidates),
            wake_candidates=len(wake_candidates)
        )

        # Process wake first (priority)
        for src_dpid, src_port, dst_dpid, dst_port in wake_candidates:
            try:
                self._wake_link(src_dpid, src_port, dst_dpid, dst_port)
            except Exception as e:
                logger.error(
                    "wake_failed",
                    src_dpid=src_dpid,
                    src_port=src_port,
                    error=str(e)
                )

        # Process sleep
        for src_dpid, src_port, dst_dpid, dst_port in sleep_candidates[:2]:  # Limit per cycle
            try:
                self._sleep_link(src_dpid, src_port, dst_dpid, dst_port)
            except Exception as e:
                logger.error(
                    "sleep_failed",
                    src_dpid=src_dpid,
                    src_port=src_port,
                    error=str(e)
                )

    def _sleep_link(self, src_dpid, src_port, dst_dpid, dst_port):
        """Put a link to sleep."""
        # Get flows on this link
        flows = self.router.get_flows_on_link(src_dpid, src_port, dst_dpid, dst_port)

        if flows:
            # Reroute flows first
            excluded = {(src_dpid, dst_dpid)}
            reroute_paths = self.router.find_reroute_paths(flows, excluded)

            if not reroute_paths:
                logger.warning(
                    "cannot_reroute_for_sleep",
                    src_dpid=src_dpid,
                    src_port=src_port
                )
                return

            # Install new paths
            for flow_id, new_path in reroute_paths.items():
                flow = self.router._flows.get(flow_id)
                if flow:
                    # Delete old flows
                    for dpid in flow.path:
                        if dpid in self.datapaths:
                            parser = self.datapaths[dpid].ofproto_parser
                            match = parser.OFPMatch(
                                eth_type=0x0800,
                                ipv4_src=flow.src_ip,
                                ipv4_dst=flow.dst_ip
                            )
                            self._delete_flow(self.datapaths[dpid], match)

                    # Install new flows
                    self._install_path_flows(
                        flow_id,
                        new_path,
                        "", "",  # MAC addresses not needed for IP match
                        flow.src_ip,
                        flow.dst_ip,
                        self.hosts.get(flow.dst_ip, (0, 1, ""))[1]
                    )

                    # Update router
                    self.router.remove_flow(flow_id)
                    self.router.install_flow(
                        flow_id, flow.src_ip, flow.dst_ip,
                        new_path, flow.bandwidth, flow.priority
                    )

        # Update energy model
        self.energy_model.set_port_sleeping(src_dpid, src_port)
        self.energy_model.set_port_sleeping(dst_dpid, dst_port)

        # Send port mod to disable port (optional - depends on switch support)
        # self._send_port_mod(src_dpid, src_port, sleep=True)

        logger.info(
            "link_put_to_sleep",
            src_dpid=src_dpid,
            src_port=src_port,
            dst_dpid=dst_dpid,
            dst_port=dst_port
        )

    def _wake_link(self, src_dpid, src_port, dst_dpid, dst_port):
        """Wake up a sleeping link."""
        self.energy_model.set_port_active(src_dpid, src_port)
        self.energy_model.set_port_active(dst_dpid, dst_port)

        # Send port mod to enable port (optional)
        # self._send_port_mod(src_dpid, src_port, sleep=False)

        logger.info(
            "link_woken_up",
            src_dpid=src_dpid,
            src_port=src_port,
            dst_dpid=dst_dpid,
            dst_port=dst_port
        )

    async def _async_install_path(self, flow_id, src_ip, dst_ip, path, links):
        """Async callback for flow installation."""
        pass  # Handled synchronously in _sleep_link

    async def _async_port_mod(self, dpid, port, sleep=True):
        """Async callback for port modification."""
        pass  # Handled synchronously

    def get_network_stats(self) -> Dict:
        """Get comprehensive network statistics for dashboard."""
        energy_stats = self.energy_model.get_stats()
        router_stats = self.router.get_stats()
        predictor_stats = self.predictor.get_stats()
        sleep_stats = self.sleep_manager.get_stats()
        collector_stats = self.stats_collector.get_stats()
        qos_metrics = self.stats_collector.get_qos_metrics()
        ecmp_comparison = self.stats_collector.get_ecmp_comparison()

        return {
            "timestamp": time.time(),
            "energy": energy_stats,
            "routing": router_stats,
            "prediction": predictor_stats,
            "sleep_manager": sleep_stats,
            "collector": collector_stats,
            "qos": qos_metrics,
            "ecmp_comparison": ecmp_comparison,
            "datapaths": list(self.datapaths.keys()),
            "hosts": len(self.hosts)
        }

    def get_topology(self) -> Dict:
        """Get topology information for visualization."""
        return self.router.get_topology_info()

    def close(self):
        """Cleanup on shutdown."""
        self._running = False
        logger.info("ecoroute_controller_shutdown")
