"""
Enhanced Greedy Energy-Aware Router for EcoRoute SDN Controller

Implements energy-aware path selection using:
- Yen's K-shortest paths algorithm
- Energy cost scoring based on port sleep states
- Load-aware path selection with QoS constraints
- Path consolidation to maximize sleeping links
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import structlog

from controller.energy_model import EnergyModel, PortState
from controller.ewma_predictor import AdaptiveEWMAPredictor

logger = structlog.get_logger(__name__)


@dataclass
class PathScore:
    """Scoring result for a path."""
    path: List[int]  # List of node IDs (dpids)
    links: List[Tuple[int, int, int, int]]  # (src_dpid, src_port, dst_dpid, dst_port)
    total_score: float
    energy_score: float
    load_score: float
    hop_score: float
    sleeping_links_used: int
    active_links_used: int
    max_utilization: float


@dataclass
class Flow:
    """Represents an active flow in the network."""
    flow_id: str
    src_ip: str
    dst_ip: str
    src_dpid: int
    dst_dpid: int
    path: List[int]
    links: List[Tuple[int, int, int, int]]
    bandwidth: float  # Mbps
    priority: int = 0
    created_at: float = field(default_factory=time.time)


class EnergyAwareRouter:
    """
    Enhanced Greedy Energy-Aware Router

    Features:
    - K-shortest paths using Yen's algorithm
    - Energy-aware path scoring (prefer paths using active links)
    - Load balancing with predicted utilization
    - QoS constraint checking (max utilization, latency)
    - Flow consolidation for maximizing sleep opportunities
    """

    def __init__(
        self,
        energy_model: EnergyModel,
        predictor: AdaptiveEWMAPredictor,
        k_paths: int = 3,
        energy_weight: float = 0.5,
        load_weight: float = 0.3,
        hop_weight: float = 0.2,
        hop_penalty: float = 0.1,
        max_utilization: float = 80.0
    ):
        """
        Initialize Energy-Aware Router.

        Args:
            energy_model: Energy model for power state tracking
            predictor: EWMA predictor for load forecasting
            k_paths: Number of shortest paths to consider
            energy_weight: Weight for energy cost in scoring
            load_weight: Weight for load cost in scoring
            hop_weight: Weight for hop count in scoring
            hop_penalty: Penalty per hop
            max_utilization: Maximum allowed link utilization (%)
        """
        self.energy_model = energy_model
        self.predictor = predictor
        self.k_paths = k_paths
        self.energy_weight = energy_weight
        self.load_weight = load_weight
        self.hop_weight = hop_weight
        self.hop_penalty = hop_penalty
        self.max_utilization = max_utilization

        # Network graph
        self._graph: nx.DiGraph = nx.DiGraph()

        # Link information: (src_dpid, dst_dpid) -> (src_port, dst_port, capacity)
        self._link_info: Dict[Tuple[int, int], Tuple[int, int, float]] = {}

        # Link utilization tracking
        self._link_utilization: Dict[Tuple[int, int], float] = defaultdict(float)

        # Active flows
        self._flows: Dict[str, Flow] = {}

        # Host to switch mapping: host_ip -> (dpid, port)
        self._host_map: Dict[str, Tuple[int, int]] = {}

        logger.info(
            "energy_router_initialized",
            k_paths=k_paths,
            energy_weight=energy_weight,
            load_weight=load_weight,
            max_utilization=max_utilization
        )

    def add_link(
        self,
        src_dpid: int,
        src_port: int,
        dst_dpid: int,
        dst_port: int,
        capacity: float = 1000.0  # Mbps
    ):
        """
        Add a link to the network topology.

        Args:
            src_dpid: Source switch datapath ID
            src_port: Source port number
            dst_dpid: Destination switch datapath ID
            dst_port: Destination port number
            capacity: Link capacity in Mbps
        """
        self._graph.add_edge(
            src_dpid,
            dst_dpid,
            src_port=src_port,
            dst_port=dst_port,
            capacity=capacity,
            weight=1.0
        )
        self._link_info[(src_dpid, dst_dpid)] = (src_port, dst_port, capacity)

        logger.debug(
            "link_added",
            src_dpid=src_dpid,
            src_port=src_port,
            dst_dpid=dst_dpid,
            dst_port=dst_port,
            capacity=capacity
        )

    def remove_link(self, src_dpid: int, dst_dpid: int):
        """Remove a link from the topology."""
        if self._graph.has_edge(src_dpid, dst_dpid):
            self._graph.remove_edge(src_dpid, dst_dpid)
            self._link_info.pop((src_dpid, dst_dpid), None)
            logger.debug("link_removed", src_dpid=src_dpid, dst_dpid=dst_dpid)

    def add_host(self, host_ip: str, dpid: int, port: int):
        """Register a host's location."""
        self._host_map[host_ip] = (dpid, port)
        logger.debug("host_added", host_ip=host_ip, dpid=dpid, port=port)

    def remove_host(self, host_ip: str):
        """Remove a host from the topology."""
        self._host_map.pop(host_ip, None)

    def get_host_location(self, host_ip: str) -> Optional[Tuple[int, int]]:
        """Get the switch and port where a host is connected."""
        return self._host_map.get(host_ip)

    def update_link_utilization(
        self,
        src_dpid: int,
        dst_dpid: int,
        utilization: float
    ):
        """Update the current utilization of a link."""
        self._link_utilization[(src_dpid, dst_dpid)] = utilization

    def find_k_shortest_paths(
        self,
        src_dpid: int,
        dst_dpid: int,
        k: Optional[int] = None
    ) -> List[List[int]]:
        """
        Find k-shortest paths using Yen's algorithm.

        Args:
            src_dpid: Source switch
            dst_dpid: Destination switch
            k: Number of paths (defaults to self.k_paths)

        Returns:
            List of paths, where each path is a list of dpids
        """
        if k is None:
            k = self.k_paths

        if not self._graph.has_node(src_dpid) or not self._graph.has_node(dst_dpid):
            logger.warning(
                "path_not_found_missing_nodes",
                src_dpid=src_dpid,
                dst_dpid=dst_dpid
            )
            return []

        try:
            # Use NetworkX's implementation of Yen's k-shortest paths
            paths = list(
                nx.shortest_simple_paths(
                    self._graph,
                    src_dpid,
                    dst_dpid,
                    weight='weight'
                )
            )[:k]
            return paths
        except nx.NetworkXNoPath:
            logger.warning(
                "no_path_exists",
                src_dpid=src_dpid,
                dst_dpid=dst_dpid
            )
            return []
        except Exception as e:
            logger.error(
                "path_finding_error",
                src_dpid=src_dpid,
                dst_dpid=dst_dpid,
                error=str(e)
            )
            return []

    def _path_to_links(
        self,
        path: List[int]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Convert a path (list of dpids) to links with port information.

        Returns:
            List of (src_dpid, src_port, dst_dpid, dst_port) tuples
        """
        links = []
        for i in range(len(path) - 1):
            src_dpid = path[i]
            dst_dpid = path[i + 1]
            link_info = self._link_info.get((src_dpid, dst_dpid))
            if link_info:
                src_port, dst_port, _ = link_info
                links.append((src_dpid, src_port, dst_dpid, dst_port))
        return links

    def _calculate_energy_score(
        self,
        links: List[Tuple[int, int, int, int]]
    ) -> Tuple[float, int, int]:
        """
        Calculate energy score for a path.

        Lower score = better (prefers active links over sleeping ones).

        Returns:
            (energy_score, sleeping_links_used, active_links_used)
        """
        if not links:
            return 0.0, 0, 0

        total_cost = 0.0
        sleeping_count = 0
        active_count = 0

        for src_dpid, src_port, dst_dpid, dst_port in links:
            link_cost = self.energy_model.get_link_energy_cost(
                src_dpid, src_port, dst_dpid, dst_port
            )
            total_cost += link_cost

            # Count sleeping vs active
            if self.energy_model.is_port_sleeping(src_dpid, src_port):
                sleeping_count += 1
            else:
                active_count += 1

            if self.energy_model.is_port_sleeping(dst_dpid, dst_port):
                sleeping_count += 1
            else:
                active_count += 1

        # Normalize by number of links
        energy_score = total_cost / len(links) if links else 0.0

        return energy_score, sleeping_count, active_count

    def _calculate_load_score(
        self,
        path: List[int],
        additional_bandwidth: float = 0.0
    ) -> Tuple[float, float]:
        """
        Calculate load score based on predicted utilization.

        Lower score = better (prefers less loaded paths).

        Returns:
            (load_score, max_utilization)
        """
        if len(path) < 2:
            return 0.0, 0.0

        max_util = 0.0
        total_util = 0.0

        for i in range(len(path) - 1):
            src_dpid = path[i]
            dst_dpid = path[i + 1]

            # Get current utilization
            current_util = self._link_utilization.get((src_dpid, dst_dpid), 0.0)

            # Get predicted utilization
            link_info = self._link_info.get((src_dpid, dst_dpid))
            if link_info:
                src_port, _, capacity = link_info
                prediction = self.predictor.get_prediction(src_dpid, src_port)
                if prediction:
                    predicted_util = prediction.predicted_load
                else:
                    predicted_util = current_util

                # Add bandwidth requirement
                if capacity > 0:
                    predicted_util += (additional_bandwidth / capacity) * 100

                max_util = max(max_util, predicted_util)
                total_util += predicted_util

        # Average utilization as score
        load_score = total_util / (len(path) - 1) if len(path) > 1 else 0.0

        # Normalize to 0-1 range
        load_score = min(1.0, load_score / 100.0)

        return load_score, max_util

    def _calculate_hop_score(self, path: List[int]) -> float:
        """
        Calculate hop count penalty.

        Higher penalty for longer paths.
        """
        hop_count = len(path) - 1
        return hop_count * self.hop_penalty

    def score_path(
        self,
        path: List[int],
        bandwidth_required: float = 0.0
    ) -> PathScore:
        """
        Score a path based on energy, load, and hop count.

        Args:
            path: List of dpids
            bandwidth_required: Bandwidth needed for the flow (Mbps)

        Returns:
            PathScore with detailed scoring breakdown
        """
        links = self._path_to_links(path)

        energy_score, sleeping_used, active_used = self._calculate_energy_score(links)
        load_score, max_util = self._calculate_load_score(path, bandwidth_required)
        hop_score = self._calculate_hop_score(path)

        # Combined weighted score (lower is better)
        total_score = (
            self.energy_weight * energy_score +
            self.load_weight * load_score +
            self.hop_weight * hop_score
        )

        return PathScore(
            path=path,
            links=links,
            total_score=total_score,
            energy_score=energy_score,
            load_score=load_score,
            hop_score=hop_score,
            sleeping_links_used=sleeping_used,
            active_links_used=active_used,
            max_utilization=max_util
        )

    def find_best_path(
        self,
        src_dpid: int,
        dst_dpid: int,
        bandwidth_required: float = 0.0
    ) -> Optional[PathScore]:
        """
        Find the best energy-aware path between two switches.

        Args:
            src_dpid: Source switch
            dst_dpid: Destination switch
            bandwidth_required: Required bandwidth (Mbps)

        Returns:
            Best PathScore or None if no valid path exists
        """
        paths = self.find_k_shortest_paths(src_dpid, dst_dpid)

        if not paths:
            return None

        # Score all paths
        scored_paths = []
        for path in paths:
            score = self.score_path(path, bandwidth_required)

            # Check QoS constraints
            if score.max_utilization > self.max_utilization:
                logger.debug(
                    "path_rejected_qos",
                    path=path,
                    max_util=score.max_utilization,
                    threshold=self.max_utilization
                )
                continue

            scored_paths.append(score)

        if not scored_paths:
            # All paths violate QoS - return best anyway with warning
            logger.warning(
                "all_paths_exceed_qos",
                src_dpid=src_dpid,
                dst_dpid=dst_dpid,
                threshold=self.max_utilization
            )
            scored_paths = [self.score_path(p, bandwidth_required) for p in paths]

        # Select best path (lowest score)
        best = min(scored_paths, key=lambda s: s.total_score)

        logger.info(
            "best_path_selected",
            src_dpid=src_dpid,
            dst_dpid=dst_dpid,
            path=best.path,
            score=round(best.total_score, 3),
            energy_score=round(best.energy_score, 3),
            load_score=round(best.load_score, 3),
            sleeping_links=best.sleeping_links_used,
            active_links=best.active_links_used
        )

        return best

    def install_flow(
        self,
        flow_id: str,
        src_ip: str,
        dst_ip: str,
        path_score: PathScore,
        bandwidth: float = 0.0,
        priority: int = 0
    ) -> Flow:
        """
        Register a flow installation.

        Args:
            flow_id: Unique flow identifier
            src_ip: Source IP address
            dst_ip: Destination IP address
            path_score: Selected path scoring
            bandwidth: Flow bandwidth (Mbps)
            priority: Flow priority

        Returns:
            Created Flow object
        """
        flow = Flow(
            flow_id=flow_id,
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_dpid=path_score.path[0],
            dst_dpid=path_score.path[-1],
            path=path_score.path,
            links=path_score.links,
            bandwidth=bandwidth,
            priority=priority
        )

        self._flows[flow_id] = flow

        # Update link utilization estimates
        for src, dst in zip(path_score.path[:-1], path_score.path[1:]):
            link_info = self._link_info.get((src, dst))
            if link_info:
                _, _, capacity = link_info
                current = self._link_utilization.get((src, dst), 0.0)
                added = (bandwidth / capacity) * 100 if capacity > 0 else 0
                self._link_utilization[(src, dst)] = current + added

        logger.info(
            "flow_installed",
            flow_id=flow_id,
            src_ip=src_ip,
            dst_ip=dst_ip,
            path=path_score.path,
            bandwidth=bandwidth
        )

        return flow

    def remove_flow(self, flow_id: str) -> Optional[Flow]:
        """
        Remove a flow and update utilization tracking.

        Returns:
            Removed Flow or None if not found
        """
        flow = self._flows.pop(flow_id, None)

        if flow:
            # Update link utilization
            for src, dst in zip(flow.path[:-1], flow.path[1:]):
                link_info = self._link_info.get((src, dst))
                if link_info:
                    _, _, capacity = link_info
                    current = self._link_utilization.get((src, dst), 0.0)
                    removed = (flow.bandwidth / capacity) * 100 if capacity > 0 else 0
                    self._link_utilization[(src, dst)] = max(0, current - removed)

            logger.info("flow_removed", flow_id=flow_id)

        return flow

    def get_flows_on_link(
        self,
        src_dpid: int,
        src_port: int,
        dst_dpid: int,
        dst_port: int
    ) -> List[Flow]:
        """Get all flows traversing a specific link."""
        flows = []
        for flow in self._flows.values():
            for link in flow.links:
                if link == (src_dpid, src_port, dst_dpid, dst_port):
                    flows.append(flow)
                    break
        return flows

    def can_reroute_flows(
        self,
        link_flows: List[Flow],
        src_dpid: int,
        dst_dpid: int
    ) -> bool:
        """
        Check if flows on a link can be rerouted to allow link sleep.

        Args:
            link_flows: Flows currently on the link
            src_dpid: Link source
            dst_dpid: Link destination

        Returns:
            True if all flows can be rerouted
        """
        for flow in link_flows:
            # Create temporary graph without the link to sleep
            temp_graph = self._graph.copy()
            if temp_graph.has_edge(src_dpid, dst_dpid):
                temp_graph.remove_edge(src_dpid, dst_dpid)

            # Check if alternate path exists
            try:
                alt_path = nx.shortest_path(
                    temp_graph,
                    flow.src_dpid,
                    flow.dst_dpid,
                    weight='weight'
                )

                # Verify QoS on alternate path
                score = self.score_path(alt_path, flow.bandwidth)
                if score.max_utilization > self.max_utilization:
                    return False

            except nx.NetworkXNoPath:
                return False

        return True

    def find_reroute_paths(
        self,
        link_flows: List[Flow],
        excluded_links: Set[Tuple[int, int]]
    ) -> Dict[str, PathScore]:
        """
        Find reroute paths for flows, avoiding excluded links.

        Args:
            link_flows: Flows to reroute
            excluded_links: Links to avoid (src_dpid, dst_dpid)

        Returns:
            Dict mapping flow_id to new PathScore, empty if rerouting impossible
        """
        reroute_paths = {}

        # Create graph without excluded links
        temp_graph = self._graph.copy()
        for src, dst in excluded_links:
            if temp_graph.has_edge(src, dst):
                temp_graph.remove_edge(src, dst)

        for flow in link_flows:
            try:
                paths = list(
                    nx.shortest_simple_paths(
                        temp_graph,
                        flow.src_dpid,
                        flow.dst_dpid,
                        weight='weight'
                    )
                )[:self.k_paths]

                if not paths:
                    return {}

                # Score and select best
                best = None
                for path in paths:
                    score = self.score_path(path, flow.bandwidth)
                    if score.max_utilization <= self.max_utilization:
                        if best is None or score.total_score < best.total_score:
                            best = score

                if best is None:
                    return {}

                reroute_paths[flow.flow_id] = best

            except nx.NetworkXNoPath:
                return {}

        return reroute_paths

    def get_links_to_sleep(self, min_flows: int = 0) -> List[Tuple[int, int]]:
        """
        Identify links that are candidates for sleeping.

        Returns links that:
        - Have low predicted utilization
        - Can have their flows rerouted
        - Are not critical (have alternate paths)

        Returns:
            List of (src_dpid, dst_dpid) tuples
        """
        candidates = []

        for (src_dpid, dst_dpid), (src_port, dst_port, _) in self._link_info.items():
            # Skip if already sleeping
            if self.energy_model.is_port_sleeping(src_dpid, src_port):
                continue

            # Check predicted utilization
            if not self.predictor.should_sleep(src_dpid, src_port):
                continue

            # Get flows on this link
            flows = self.get_flows_on_link(src_dpid, src_port, dst_dpid, dst_port)

            if len(flows) < min_flows:
                continue

            # Check if flows can be rerouted
            if flows and not self.can_reroute_flows(flows, src_dpid, dst_dpid):
                continue

            candidates.append((src_dpid, dst_dpid))

        return candidates

    def get_topology_info(self) -> Dict:
        """Get topology information for dashboard."""
        nodes = list(self._graph.nodes())
        edges = []

        for src, dst in self._graph.edges():
            link_info = self._link_info.get((src, dst))
            if link_info:
                src_port, dst_port, capacity = link_info
                sleeping = (
                    self.energy_model.is_port_sleeping(src, src_port) or
                    self.energy_model.is_port_sleeping(dst, dst_port)
                )
                util = self._link_utilization.get((src, dst), 0.0)

                edges.append({
                    "source": src,
                    "target": dst,
                    "src_port": src_port,
                    "dst_port": dst_port,
                    "capacity": capacity,
                    "utilization": round(util, 2),
                    "sleeping": sleeping
                })

        return {
            "nodes": nodes,
            "edges": edges,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "active_flows": len(self._flows),
            "hosts": dict(self._host_map)
        }

    def get_stats(self) -> Dict:
        """Get router statistics."""
        total_links = len(self._link_info)
        sleeping_links = sum(
            1 for (src, dst), (src_port, _, _) in self._link_info.items()
            if self.energy_model.is_port_sleeping(src, src_port)
        )

        avg_util = (
            sum(self._link_utilization.values()) / len(self._link_utilization)
            if self._link_utilization else 0.0
        )

        return {
            "total_nodes": self._graph.number_of_nodes(),
            "total_links": total_links,
            "active_links": total_links - sleeping_links,
            "sleeping_links": sleeping_links,
            "active_flows": len(self._flows),
            "total_hosts": len(self._host_map),
            "average_link_utilization": round(avg_util, 2),
            "k_paths": self.k_paths
        }

    def reset(self):
        """Reset router state."""
        self._graph.clear()
        self._link_info.clear()
        self._link_utilization.clear()
        self._flows.clear()
        self._host_map.clear()
        logger.info("energy_router_reset")
