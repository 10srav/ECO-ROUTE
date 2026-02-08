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

import json
import os
import sys
import time
from pathlib import Path
from threading import Thread
from typing import Dict, List, Optional, Set, Tuple

import yaml
from webob import Response

# Use compatibility shim so both os-ken and ryu work
from controller.compat import (
    app_manager,
    ofp_event,
    CONFIG_DISPATCHER,
    DEAD_DISPATCHER,
    MAIN_DISPATCHER,
    set_ev_cls,
    hub,
    ofproto_v1_3,
    topo_event,
    get_all_link,
    get_all_switch,
    ControllerBase,
    WSGIApplication,
    route,
)
from controller.compat import (
    packet_lib as packet_module,
    ethernet as ethernet_mod,
    arp as arp_mod,
    ipv4 as ipv4_mod,
    icmp as icmp_mod,
    tcp as tcp_mod,
    udp as udp_mod,
    lldp as lldp_mod,
)

# Re-bind packet classes to match the original import style
# (from os_ken.lib.packet import arp, ethernet, ..., packet)
packet = packet_module
ethernet = ethernet_mod
arp = arp_mod
ipv4 = ipv4_mod
icmp = icmp_mod
tcp = tcp_mod
udp = udp_mod
lldp = lldp_mod

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


class EcoRouteController(app_manager.OSKenApp):
    """
    EcoRoute Energy-Aware SDN Controller

    Main Ryu application that coordinates all EcoRoute modules:
    - Traffic prediction (EWMA)
    - Energy-aware routing
    - Link sleep/wake management
    - Statistics collection
    - REST API for dashboard (via WSGI on port 8080)
    """

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {'wsgi': WSGIApplication}
    _LLDP_MAGIC = b'\xec\x01'  # Magic prefix for EcoRoute LLDP frames

    def __init__(self, *args, **kwargs):
        super(EcoRouteController, self).__init__(*args, **kwargs)

        # Register REST API
        wsgi = kwargs['wsgi']
        wsgi.register(EcoRouteRestAPI, {'ecoroute_app': self})

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

        # Set callbacks (synchronous - compatible with eventlet green threads)
        self.sleep_manager.set_flow_mod_callback(self._sync_install_path)
        self.sleep_manager.set_port_mod_callback(self._sync_port_mod)
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

        # Track discovered links to avoid duplicate add_link calls
        self._discovered_links: Set[Tuple[int, int, int, int]] = set()

        # Flood dedup cache: (src_mac, dst_identifier) -> timestamp
        # Prevents broadcast storms when flooding unknown destinations
        self._flood_cache: Dict[Tuple[str, str], float] = {}
        self._FLOOD_TIMEOUT = 2.0  # seconds

        # Start background threads
        self.stats_thread = hub.spawn(self._stats_polling_loop)
        self.optimization_thread = hub.spawn(self._optimization_loop)
        self.lldp_thread = hub.spawn(self._lldp_send_loop)

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

                # Clean up router state: remove links involving this switch
                links_to_remove = [
                    (src, dst) for (src, dst) in self.router._link_info
                    if src == dpid or dst == dpid
                ]
                for src, dst in links_to_remove:
                    self.router.remove_link(src, dst)

                # Clean up MAC table for this switch
                self.mac_to_port.pop(dpid, None)

                # Clean up hosts connected to this switch
                hosts_to_remove = [
                    ip for ip, (sw, port, mac) in self.hosts.items()
                    if sw == dpid
                ]
                for ip in hosts_to_remove:
                    del self.hosts[ip]
                    self.arp_table.pop(ip, None)
                    self.router.remove_host(ip)

                logger.info("switch_disconnected", dpid=dpid, removed_links=len(links_to_remove), removed_hosts=len(hosts_to_remove))

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

        try:
            pkt = packet.Packet(msg.data)
            eth = pkt.get_protocol(ethernet.ethernet)
        except (UnicodeDecodeError, Exception) as e:
            # Binary/malformed packets can cause decode errors in ryu/os-ken
            logger.debug("packet_parse_error", dpid=dpid, in_port=in_port, error=str(e))
            return

        if eth is None:
            return

        if eth.ethertype == 0x88cc:  # LLDP — parse for link discovery
            self._handle_lldp(dpid, in_port, pkt)
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

        # Non-ARP/non-IPv4: drop to prevent broadcast storms
        # (LLDP is already filtered above)
        logger.debug(
            "unhandled_ethertype_dropped",
            dpid=dpid,
            in_port=in_port,
            ethertype=hex(eth.ethertype)
        )
        return

    def _handle_arp(self, datapath, in_port, eth, arp_pkt, data):
        """Handle ARP packets for host discovery."""
        dpid = datapath.id
        src_ip = arp_pkt.src_ip
        src_mac = arp_pkt.src_mac

        # Don't process ARP that arrived from a switch-facing port
        # (it was flooded from another switch — processing it would corrupt host table)
        switch_ports = self._get_switch_facing_ports(dpid)
        if in_port in switch_ports:
            return

        # Register host — allow updates from any edge switch (host mobility)
        # Only skip if the same switch+port is already recorded (no change)
        existing = self.hosts.get(src_ip)
        if existing is None or existing != (dpid, in_port, src_mac):
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
                # Reply on behalf of destination (proxy ARP)
                self._send_arp_reply(
                    datapath, in_port, eth,
                    dst_ip, self.arp_table[dst_ip],
                    src_ip, src_mac
                )
            else:
                # Deduplicate ARP floods to prevent broadcast storms
                flood_key = (src_mac, dst_ip)
                now = time.time()
                if flood_key in self._flood_cache and \
                        now - self._flood_cache[flood_key] < self._FLOOD_TIMEOUT:
                    return  # Already flooded this ARP recently
                self._flood_cache[flood_key] = now
                self._flood_to_hosts(datapath, in_port, data)

        elif arp_pkt.opcode == arp.ARP_REPLY:
            # Forward ARP reply to the destination host
            dst_ip = arp_pkt.dst_ip
            if dst_ip in self.hosts:
                dst_dpid, dst_port, _ = self.hosts[dst_ip]
                if dst_dpid in self.datapaths:
                    self._send_packet(
                        self.datapaths[dst_dpid], dst_port, data
                    )

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

        # Check if packet arrived from a switch-facing port (was flooded/forwarded)
        switch_ports = self._get_switch_facing_ports(dpid)
        from_switch = in_port in switch_ports

        # Register source host if not known (only from host-facing ports)
        if not from_switch and src_ip not in self.hosts:
            self.hosts[src_ip] = (dpid, in_port, eth.src)
            self.router.add_host(src_ip, dpid, in_port)

        # If packet came from a switch port (flooded), only deliver locally
        if from_switch:
            if dst_ip in self.hosts:
                dst_dpid, dst_port, dst_mac = self.hosts[dst_ip]
                if dpid == dst_dpid:
                    self._install_direct_flow(datapath, eth, ip_pkt, dst_port)
                    self._send_packet(datapath, dst_port, data)
            return

        # Check if destination is known
        if dst_ip not in self.hosts:
            # Destination unknown — flood to discover host (instead of dropping)
            flood_key = (eth.src, eth.dst)
            now = time.time()
            if flood_key in self._flood_cache and \
                    now - self._flood_cache[flood_key] < self._FLOOD_TIMEOUT:
                return  # Already flooded recently
            self._flood_cache[flood_key] = now
            # Periodic cleanup of stale entries
            if len(self._flood_cache) > 200:
                self._flood_cache = {
                    k: v for k, v in self._flood_cache.items()
                    if now - v < self._FLOOD_TIMEOUT
                }
            self._flood_to_hosts(datapath, in_port, data)
            return

        dst_dpid, dst_port, dst_mac = self.hosts[dst_ip]

        # Same switch - direct forward
        if dpid == dst_dpid:
            self._install_direct_flow(datapath, eth, ip_pkt, dst_port)
            self._send_packet(datapath, dst_port, data)
            return

        # Find energy-aware path
        path_score = self.router.find_best_path(dpid, dst_dpid)

        if path_score and path_score.sleeping_links_used > 0:
            # On-demand wake: traffic needs links that are sleeping
            for lsrc, lsrc_port, ldst, ldst_port in path_score.links:
                if (self.energy_model.is_port_sleeping(lsrc, lsrc_port) or
                        self.energy_model.is_port_sleeping(ldst, ldst_port)):
                    self._wake_link(lsrc, lsrc_port, ldst, ldst_port)
                    self.predictor.reset(lsrc, lsrc_port)
                    self.predictor.reset(ldst, ldst_port)
            # Wait for ports to become operational before installing flows
            hub.sleep(self.energy_model._wake_latency_ms / 1000.0)
            logger.info(
                "on_demand_wake",
                src_dpid=dpid,
                dst_dpid=dst_dpid,
                sleeping_links_woken=path_score.sleeping_links_used
            )

        if not path_score:
            logger.warning(
                "no_path_found",
                src_dpid=dpid,
                dst_dpid=dst_dpid
            )
            # Drop - no valid path exists; host will retry after topology converges
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

    def _get_switch_facing_ports(self, dpid):
        """Get set of ports on this switch that connect to other switches."""
        switch_ports = set()
        for (s, d), (sp, dp, _) in self.router._link_info.items():
            if s == dpid:
                switch_ports.add(sp)
            if d == dpid:
                switch_ports.add(dp)
        return switch_ports

    def _get_host_facing_ports(self, dpid):
        """Get ports that face hosts (not connected to other switches)."""
        switch = self.energy_model._switches.get(dpid)
        if not switch:
            return set()
        all_ports = set(switch.ports.keys())
        switch_ports = self._get_switch_facing_ports(dpid)
        return all_ports - switch_ports

    def _flood_to_hosts(self, source_datapath, in_port, data):
        """Flood packet to host-facing ports only, preventing broadcast storms.

        Instead of OFPP_FLOOD (which causes loops in multi-switch topologies),
        send individual PacketOut messages to each host-facing port on every
        edge switch. Core/aggregation switches have no host ports and are skipped.
        """
        source_dpid = source_datapath.id
        sent = False

        for dpid, dp in self.datapaths.items():
            host_ports = self._get_host_facing_ports(dpid)
            if not host_ports:
                continue

            parser = dp.ofproto_parser
            for port in host_ports:
                if dpid == source_dpid and port == in_port:
                    continue
                actions = [parser.OFPActionOutput(port)]
                out = parser.OFPPacketOut(
                    datapath=dp,
                    buffer_id=dp.ofproto.OFP_NO_BUFFER,
                    in_port=dp.ofproto.OFPP_CONTROLLER,
                    actions=actions,
                    data=data
                )
                dp.send_msg(out)
                sent = True

        if not sent:
            # Fallback: topology not yet discovered, flood on entry switch only
            logger.debug("flood_fallback_no_topology", dpid=source_dpid)
            ofproto = source_datapath.ofproto
            parser = source_datapath.ofproto_parser
            actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
            out = parser.OFPPacketOut(
                datapath=source_datapath,
                buffer_id=ofproto.OFP_NO_BUFFER,
                in_port=in_port,
                actions=actions,
                data=data
            )
            source_datapath.send_msg(out)

    def _handle_lldp(self, dst_dpid, dst_port, pkt):
        """Parse LLDP packet to discover a link between two switches."""
        lldp_pkt = pkt.get_protocol(lldp.lldp)
        if not lldp_pkt or len(lldp_pkt.tlvs) < 2:
            return

        # TLV 0: Chassis ID = magic + dpid as binary, TLV 1: Port ID = port_no
        try:
            chassis_tlv = lldp_pkt.tlvs[0]
            port_tlv = lldp_pkt.tlvs[1]

            # Validate magic prefix to reject external/foreign LLDP frames
            chassis_data = chassis_tlv.chassis_id
            if len(chassis_data) < 10 or chassis_data[:2] != self._LLDP_MAGIC:
                logger.debug(
                    "lldp_rejected_no_magic",
                    raw_chassis=chassis_data.hex() if chassis_data else "empty"
                )
                return

            src_dpid = int.from_bytes(chassis_data[2:], 'big')
            src_port = int.from_bytes(port_tlv.port_id, 'big')
        except Exception as e:
            logger.debug("lldp_parse_failed", error=str(e))
            return

        # Validate src_dpid is a known switch managed by this controller
        if src_dpid not in self.datapaths:
            logger.debug(
                "lldp_rejected_unknown_dpid",
                src_dpid=src_dpid,
                src_port=src_port
            )
            return

        if src_dpid == dst_dpid:
            return  # Ignore self-loops

        link_key = (src_dpid, src_port, dst_dpid, dst_port)
        if link_key in self._discovered_links:
            return  # Already known

        self._discovered_links.add(link_key)

        # Register link in router and energy model
        self.router.add_link(src_dpid, src_port, dst_dpid, dst_port, 1000.0)

        logger.info(
            "link_discovered",
            src_dpid=src_dpid, src_port=src_port,
            dst_dpid=dst_dpid, dst_port=dst_port
        )

    def _lldp_send_loop(self):
        """Periodically send LLDP packets out every port for link discovery."""
        # Wait briefly for switches to connect and report ports
        hub.sleep(2)

        while self._running:
            for dpid, datapath in list(self.datapaths.items()):
                switch = self.energy_model._switches.get(dpid)
                if not switch:
                    continue
                for port_no in switch.ports:
                    self._send_lldp(datapath, port_no)
            hub.sleep(3)  # Send LLDP every 3 seconds

    def _send_lldp(self, datapath, port_no):
        """Send an LLDP packet out of a specific port."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        dpid = datapath.id

        # Generate locally-administered MAC from dpid (bit 1 of octet 0 = local)
        src_mac = '02:ec:%02x:%02x:%02x:%02x' % (
            (dpid >> 24) & 0xff, (dpid >> 16) & 0xff,
            (dpid >> 8) & 0xff, dpid & 0xff
        )

        # Build LLDP packet with dpid and port_no encoded
        pkt = packet.Packet()
        pkt.add_protocol(ethernet.ethernet(
            ethertype=0x88cc,
            src=src_mac,
            dst=lldp.LLDP_MAC_NEAREST_BRIDGE
        ))

        chassis_id = lldp.ChassisID(
            subtype=lldp.ChassisID.SUB_LOCALLY_ASSIGNED,
            chassis_id=self._LLDP_MAGIC + dpid.to_bytes(8, 'big')
        )
        port_id = lldp.PortID(
            subtype=lldp.PortID.SUB_LOCALLY_ASSIGNED,
            port_id=port_no.to_bytes(4, 'big')
        )
        ttl = lldp.TTL(ttl=120)
        end = lldp.End()

        pkt.add_protocol(lldp.lldp(tlvs=[chassis_id, port_id, ttl, end]))
        pkt.serialize()

        actions = [parser.OFPActionOutput(port_no)]
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=ofproto.OFP_NO_BUFFER,
            in_port=ofproto.OFPP_CONTROLLER,
            actions=actions,
            data=pkt.data
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
            # Convert match to dict using to_jsondict
            try:
                if hasattr(stat, 'match') and hasattr(stat.match, 'to_jsondict'):
                    match_dict = stat.match.to_jsondict()
                else:
                    match_dict = {}
            except Exception:
                match_dict = {}

            flow_stats.append(FlowStats(
                dpid=dpid,
                table_id=stat.table_id,
                match=match_dict,
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

        # Set ECMP baseline from current energy state (all ports active)
        try:
            energy_stats = self.energy_model.get_stats()
            baseline_power = energy_stats.get("baseline_power_watts", 0)
            if baseline_power > 0:
                self.stats_collector.set_ecmp_baseline(baseline_power)
                logger.info("ecmp_baseline_set", baseline_power=baseline_power)
        except Exception as e:
            logger.warning("ecmp_baseline_set_failed", error=str(e))

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
                # Reset predictor for woken ports so stale EWMA data doesn't
                # immediately trigger re-sleep. Fresh stats will rebuild predictions.
                self.predictor.reset(src_dpid, src_port)
                self.predictor.reset(dst_dpid, dst_port)
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

    def _sync_install_path(self, flow_id, src_ip, dst_ip, path, links):
        """Synchronous callback for flow installation (used by SleepManager)."""
        # Flow installation is handled synchronously in _sleep_link
        pass

    def _sync_port_mod(self, dpid, port, sleep=True):
        """Synchronous callback for port modification (used by SleepManager)."""
        # Port modification is handled synchronously in _sleep_link/_wake_link
        pass

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


class EcoRouteRestAPI(ControllerBase):
    """REST API for EcoRoute controller (exposed on port 8080 via WSGI)."""

    def __init__(self, req, link, data, **config):
        super().__init__(req, link, data, **config)
        self.ecoroute_app = data['ecoroute_app']

    def _json_response(self, data):
        """Create a JSON response with CORS headers."""
        body = json.dumps(data, default=str)
        return Response(
            content_type='application/json',
            body=body,
            charset='utf-8',
            headers={
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type',
            }
        )

    def _cors_preflight_response(self):
        """Return an empty response with CORS headers for OPTIONS preflight."""
        return Response(
            status=204,
            headers={
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type',
            }
        )

    def _error_response(self, error, status=500):
        """Create an error JSON response."""
        return self._json_response({"error": str(error), "status": status})

    @route('ecoroute', '/stats', methods=['GET', 'OPTIONS'])
    def get_stats(self, req, **kwargs):
        """Get comprehensive network statistics."""
        if req.method == 'OPTIONS':
            return self._cors_preflight_response()
        try:
            stats = self.ecoroute_app.get_network_stats()
            return self._json_response(stats)
        except Exception as e:
            logger.error("rest_api_stats_error", error=str(e))
            return self._error_response(e)

    @route('ecoroute', '/topology', methods=['GET', 'OPTIONS'])
    def get_topology(self, req, **kwargs):
        """Get topology for dashboard visualization."""
        if req.method == 'OPTIONS':
            return self._cors_preflight_response()
        try:
            topology = self.ecoroute_app.get_topology()
            return self._json_response(topology)
        except Exception as e:
            logger.error("rest_api_topology_error", error=str(e))
            return self._error_response(e)

    @route('ecoroute', '/energy', methods=['GET', 'OPTIONS'])
    def get_energy(self, req, **kwargs):
        """Get energy consumption statistics."""
        if req.method == 'OPTIONS':
            return self._cors_preflight_response()
        try:
            energy = self.ecoroute_app.energy_model.get_stats()
            return self._json_response(energy)
        except Exception as e:
            logger.error("rest_api_energy_error", error=str(e))
            return self._error_response(e)

    @route('ecoroute', '/predictions', methods=['GET', 'OPTIONS'])
    def get_predictions(self, req, **kwargs):
        """Get EWMA traffic predictions."""
        if req.method == 'OPTIONS':
            return self._cors_preflight_response()
        try:
            preds = self.ecoroute_app.predictor.get_all_predictions()
            predictions_list = []
            for link_id, pred in list(preds.items())[:20]:
                predictions_list.append({
                    "link": f"link_{link_id[0]}_{link_id[1]}",
                    "current_load": round(pred.current_load, 2),
                    "predicted_load": round(pred.predicted_load, 2),
                    "confidence": round(pred.confidence, 2),
                    "trend": pred.trend
                })
            avg_conf = (
                sum(p["confidence"] for p in predictions_list) / len(predictions_list)
                if predictions_list else 0
            )
            return self._json_response({
                "predictions": predictions_list,
                "average_confidence": round(avg_conf, 2),
                "timestamp": time.time()
            })
        except Exception as e:
            logger.error("rest_api_predictions_error", error=str(e))
            return self._error_response(e)

    @route('ecoroute', '/qos', methods=['GET', 'OPTIONS'])
    def get_qos(self, req, **kwargs):
        """Get QoS metrics."""
        if req.method == 'OPTIONS':
            return self._cors_preflight_response()
        try:
            qos = self.ecoroute_app.stats_collector.get_qos_metrics()
            return self._json_response(qos)
        except Exception as e:
            logger.error("rest_api_qos_error", error=str(e))
            return self._error_response(e)

    @route('ecoroute', '/events', methods=['GET', 'OPTIONS'])
    def get_events(self, req, **kwargs):
        """Get recent sleep/wake events."""
        if req.method == 'OPTIONS':
            return self._cors_preflight_response()
        try:
            events = self.ecoroute_app.energy_model.get_events()
            return self._json_response({
                "events": events,
                "timestamp": time.time()
            })
        except Exception as e:
            logger.error("rest_api_events_error", error=str(e))
            return self._error_response(e)

    @route('ecoroute', '/ecmp-comparison', methods=['GET', 'OPTIONS'])
    def get_ecmp_comparison(self, req, **kwargs):
        """Get ECMP baseline comparison."""
        if req.method == 'OPTIONS':
            return self._cors_preflight_response()
        try:
            ecmp = self.ecoroute_app.stats_collector.get_ecmp_comparison()
            return self._json_response(ecmp)
        except Exception as e:
            logger.error("rest_api_ecmp_error", error=str(e))
            return self._error_response(e)
