#!/usr/bin/env python3
"""
Fat-Tree Topology for EcoRoute SDN Project

Creates a k-ary fat-tree data center topology with:
- k pods
- (k/2)^2 core switches
- k aggregation switches per pod
- k edge switches per pod
- (k/2) hosts per edge switch

For k=4:
- 4 core switches
- 8 aggregation switches (4 pods x 2)
- 8 edge switches (4 pods x 2)
- 16 hosts (8 edge x 2)
- Total: 20 switches

Usage:
    sudo python3 fat_tree_topo.py [--k K] [--controller IP:PORT]
    sudo mn --custom fat_tree_topo.py --topo fattree --controller remote

Author: EcoRoute Team
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel, info, error
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.topo import Topo


class FatTreeTopo(Topo):
    """
    Fat-Tree Data Center Topology

    Standard k-ary fat-tree used in data centers:
    - Full bisection bandwidth
    - Multiple paths between any pair of hosts
    - Hierarchical structure (core, aggregation, edge)
    """

    def __init__(
        self,
        k: int = 4,
        core_bw: int = 10000,  # 10 Gbps
        agg_bw: int = 10000,   # 10 Gbps
        edge_bw: int = 1000,   # 1 Gbps
        host_bw: int = 1000,   # 1 Gbps
        core_delay: str = "1ms",
        edge_delay: str = "2ms",
        **opts
    ):
        """
        Initialize Fat-Tree Topology.

        Args:
            k: Fat-tree parameter (number of ports per switch)
            core_bw: Core link bandwidth (Mbps)
            agg_bw: Aggregation link bandwidth (Mbps)
            edge_bw: Edge link bandwidth (Mbps)
            host_bw: Host link bandwidth (Mbps)
            core_delay: Core link delay
            edge_delay: Edge link delay
        """
        self.k = k
        self.core_bw = core_bw
        self.agg_bw = agg_bw
        self.edge_bw = edge_bw
        self.host_bw = host_bw
        self.core_delay = core_delay
        self.edge_delay = edge_delay

        # Switch tracking
        self.core_switches: List[str] = []
        self.agg_switches: List[str] = []
        self.edge_switches: List[str] = []
        self.hosts_list: List[str] = []

        super().__init__(**opts)

    def build(self):
        """Build the fat-tree topology."""
        k = self.k

        info(f"*** Building Fat-Tree topology with k={k}\n")

        # Number of pods
        num_pods = k

        # Core switches: (k/2)^2
        num_core = (k // 2) ** 2

        # Switches per pod
        num_agg_per_pod = k // 2
        num_edge_per_pod = k // 2

        # Hosts per edge switch
        hosts_per_edge = k // 2

        # Create core switches
        info("*** Adding core switches\n")
        for i in range(num_core):
            switch_name = f"c{i + 1}"
            # DPID starts from 1
            dpid = f"{i + 1:016x}"
            self.addSwitch(switch_name, dpid=dpid, protocols="OpenFlow13")
            self.core_switches.append(switch_name)
            info(f"    Added {switch_name} (dpid={dpid})\n")

        # Create pods
        for pod in range(num_pods):
            info(f"*** Creating pod {pod}\n")

            # Aggregation switches
            agg_in_pod = []
            for agg_idx in range(num_agg_per_pod):
                switch_name = f"a{pod}_{agg_idx}"
                dpid_num = num_core + pod * num_agg_per_pod + agg_idx + 1
                dpid = f"{dpid_num:016x}"
                self.addSwitch(switch_name, dpid=dpid, protocols="OpenFlow13")
                self.agg_switches.append(switch_name)
                agg_in_pod.append(switch_name)
                info(f"    Added aggregation switch {switch_name}\n")

            # Edge switches
            edge_in_pod = []
            for edge_idx in range(num_edge_per_pod):
                switch_name = f"e{pod}_{edge_idx}"
                dpid_num = (num_core + num_pods * num_agg_per_pod +
                          pod * num_edge_per_pod + edge_idx + 1)
                dpid = f"{dpid_num:016x}"
                self.addSwitch(switch_name, dpid=dpid, protocols="OpenFlow13")
                self.edge_switches.append(switch_name)
                edge_in_pod.append(switch_name)
                info(f"    Added edge switch {switch_name}\n")

            # Connect aggregation to edge switches
            for agg_switch in agg_in_pod:
                for edge_switch in edge_in_pod:
                    self.addLink(
                        agg_switch,
                        edge_switch,
                        cls=TCLink,
                        bw=self.edge_bw,
                        delay=self.edge_delay
                    )

            # Connect aggregation to core switches
            for agg_idx, agg_switch in enumerate(agg_in_pod):
                # Each aggregation switch connects to k/2 core switches
                core_offset = agg_idx * (k // 2)
                for i in range(k // 2):
                    core_idx = core_offset + i
                    if core_idx < len(self.core_switches):
                        core_switch = self.core_switches[core_idx]
                        self.addLink(
                            agg_switch,
                            core_switch,
                            cls=TCLink,
                            bw=self.agg_bw,
                            delay=self.core_delay
                        )

            # Add hosts to edge switches
            for edge_idx, edge_switch in enumerate(edge_in_pod):
                for h in range(hosts_per_edge):
                    host_num = (pod * num_edge_per_pod * hosts_per_edge +
                              edge_idx * hosts_per_edge + h + 1)
                    host_name = f"h{host_num}"

                    # Generate IP: 10.pod.edge.host
                    ip = f"10.{pod}.{edge_idx}.{h + 1}/24"
                    mac = f"00:00:00:{pod:02x}:{edge_idx:02x}:{h + 1:02x}"

                    self.addHost(host_name, ip=ip, mac=mac)
                    self.hosts_list.append(host_name)
                    self.addLink(
                        edge_switch,
                        host_name,
                        cls=TCLink,
                        bw=self.host_bw,
                        delay="0.5ms"
                    )
                    info(f"    Added host {host_name} ({ip})\n")

        # Print topology summary
        total_switches = len(self.core_switches) + len(self.agg_switches) + len(self.edge_switches)
        info(f"\n*** Topology Summary:\n")
        info(f"    Core switches:        {len(self.core_switches)}\n")
        info(f"    Aggregation switches: {len(self.agg_switches)}\n")
        info(f"    Edge switches:        {len(self.edge_switches)}\n")
        info(f"    Total switches:       {total_switches}\n")
        info(f"    Total hosts:          {len(self.hosts_list)}\n")


class TrafficGenerator:
    """
    Traffic generation utilities for testing EcoRoute.

    Supports:
    - iperf3 for TCP/UDP throughput tests
    - Ping for latency tests
    - Configurable traffic patterns
    """

    def __init__(self, net: Mininet):
        self.net = net

    def run_iperf(
        self,
        src: str,
        dst: str,
        duration: int = 10,
        bandwidth: str = "100M",
        protocol: str = "tcp"
    ) -> Dict:
        """
        Run iperf3 test between two hosts.

        Args:
            src: Source host name
            dst: Destination host name
            duration: Test duration in seconds
            bandwidth: Target bandwidth (e.g., "100M", "1G")
            protocol: Protocol to use ("tcp" or "udp")

        Returns:
            Dict with throughput and other metrics
        """
        src_host = self.net.get(src)
        dst_host = self.net.get(dst)

        if not src_host or not dst_host:
            return {"error": f"Host not found: {src} or {dst}"}

        dst_ip = dst_host.IP()

        # Start iperf server on destination
        dst_host.cmd(f"iperf3 -s -D -p 5001")
        time.sleep(1)

        # Run iperf client
        if protocol == "udp":
            cmd = f"iperf3 -c {dst_ip} -p 5001 -u -b {bandwidth} -t {duration} -J"
        else:
            cmd = f"iperf3 -c {dst_ip} -p 5001 -t {duration} -J"

        output = src_host.cmd(cmd)

        # Kill server
        dst_host.cmd("pkill -f 'iperf3 -s'")

        try:
            import json
            result = json.loads(output)
            return result
        except Exception:
            return {"raw_output": output}

    def run_ping(
        self,
        src: str,
        dst: str,
        count: int = 10
    ) -> Dict:
        """Run ping test between two hosts."""
        src_host = self.net.get(src)
        dst_host = self.net.get(dst)

        if not src_host or not dst_host:
            return {"error": f"Host not found: {src} or {dst}"}

        dst_ip = dst_host.IP()
        output = src_host.cmd(f"ping -c {count} {dst_ip}")

        # Parse ping output
        lines = output.split('\n')
        stats_line = [l for l in lines if 'rtt' in l.lower() or 'round-trip' in l.lower()]

        return {
            "output": output,
            "stats": stats_line[0] if stats_line else None
        }

    def generate_web_traffic(
        self,
        src: str,
        dst: str,
        requests: int = 100,
        size: str = "1K"
    ):
        """Simulate web traffic pattern."""
        src_host = self.net.get(src)
        dst_host = self.net.get(dst)
        dst_ip = dst_host.IP()

        # Start simple HTTP server
        dst_host.cmd(f"python3 -m http.server 8080 &")
        time.sleep(1)

        # Generate requests
        for _ in range(requests):
            src_host.cmd(f"curl -s http://{dst_ip}:8080 > /dev/null &")
            time.sleep(0.1)

        dst_host.cmd("pkill -f 'http.server'")

    def generate_background_traffic(
        self,
        pairs: List[Tuple[str, str]],
        bandwidth: str = "10M",
        duration: int = 60
    ):
        """Generate background traffic on multiple host pairs."""
        for src, dst in pairs:
            src_host = self.net.get(src)
            dst_host = self.net.get(dst)
            dst_ip = dst_host.IP()

            # Start server
            dst_host.cmd(f"iperf3 -s -D -p 5001")

            # Start client in background
            src_host.cmd(f"iperf3 -c {dst_ip} -p 5001 -t {duration} -b {bandwidth} &")

    def stop_all_traffic(self):
        """Stop all running traffic generators."""
        for host in self.net.hosts:
            host.cmd("pkill -f iperf3")
            host.cmd("pkill -f 'http.server'")


def run_topology(
    k: int = 4,
    controller_ip: str = "127.0.0.1",
    controller_port: int = 6653,
    with_cli: bool = True
):
    """
    Run the fat-tree topology with EcoRoute controller.

    Args:
        k: Fat-tree parameter
        controller_ip: Controller IP address
        controller_port: Controller port
        with_cli: Whether to start CLI after setup
    """
    setLogLevel('info')

    info(f"*** Creating Fat-Tree k={k} topology\n")
    topo = FatTreeTopo(k=k)

    info(f"*** Connecting to controller at {controller_ip}:{controller_port}\n")
    controller = RemoteController(
        'c0',
        ip=controller_ip,
        port=controller_port
    )

    info("*** Creating network\n")
    net = Mininet(
        topo=topo,
        controller=controller,
        switch=OVSKernelSwitch,
        link=TCLink,
        autoSetMacs=True,
        autoStaticArp=True
    )

    info("*** Starting network\n")
    net.start()

    info("*** Waiting for controller connection\n")
    time.sleep(5)

    # Test connectivity
    info("*** Testing connectivity\n")
    hosts = net.hosts
    if len(hosts) >= 2:
        result = hosts[0].cmd(f"ping -c 1 {hosts[1].IP()}")
        if "1 received" in result:
            info("*** Connectivity test passed\n")
        else:
            info("*** Connectivity test failed\n")

    if with_cli:
        info("*** Running CLI\n")
        CLI(net)

    info("*** Stopping network\n")
    net.stop()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="EcoRoute Fat-Tree Topology for Mininet"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Fat-tree parameter (default: 4)"
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="127.0.0.1:6653",
        help="Controller address (default: 127.0.0.1:6653)"
    )
    parser.add_argument(
        "--no-cli",
        action="store_true",
        help="Don't start CLI after setup"
    )

    args = parser.parse_args()

    # Parse controller address
    controller_parts = args.controller.split(":")
    controller_ip = controller_parts[0]
    controller_port = int(controller_parts[1]) if len(controller_parts) > 1 else 6653

    run_topology(
        k=args.k,
        controller_ip=controller_ip,
        controller_port=controller_port,
        with_cli=not args.no_cli
    )


# Register topology for `mn --custom`
topos = {'fattree': FatTreeTopo}


if __name__ == "__main__":
    main()
