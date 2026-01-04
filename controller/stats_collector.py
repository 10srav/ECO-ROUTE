"""
Stats Collector Module for EcoRoute SDN Controller

Handles OpenFlow statistics polling:
- OFPFlowStats for flow-level metrics
- OFPPortStats for port-level metrics
- Calculates throughput, utilization, and packet loss
- Exports metrics to CSV for analysis
"""

from __future__ import annotations

import csv
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import structlog

from controller.ewma_predictor import AdaptiveEWMAPredictor, LinkStats

logger = structlog.get_logger(__name__)


@dataclass
class FlowStats:
    """Statistics for a single flow."""
    dpid: int
    table_id: int
    match: Dict[str, Any]
    priority: int
    byte_count: int
    packet_count: int
    duration_sec: int
    duration_nsec: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class PortStats:
    """Statistics for a single port."""
    dpid: int
    port_no: int
    rx_packets: int
    tx_packets: int
    rx_bytes: int
    tx_bytes: int
    rx_dropped: int
    tx_dropped: int
    rx_errors: int
    tx_errors: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class LinkMetrics:
    """Computed metrics for a link."""
    dpid: int
    port_no: int
    throughput_mbps: float
    utilization_percent: float
    packet_loss_percent: float
    error_rate: float
    timestamp: float


@dataclass
class NetworkSnapshot:
    """Snapshot of entire network metrics."""
    timestamp: float
    total_flows: int
    total_bytes: int
    total_packets: int
    active_ports: int
    sleeping_ports: int
    average_utilization: float
    max_utilization: float
    total_energy_watts: float
    energy_savings_percent: float


class StatsCollector:
    """
    OpenFlow Statistics Collector

    Features:
    - Periodic polling of flow and port statistics
    - Throughput and utilization calculation
    - Integration with EWMA predictor
    - Metric export to CSV
    - ECMP baseline comparison
    """

    def __init__(
        self,
        predictor: AdaptiveEWMAPredictor,
        polling_interval: float = 5.0,
        export_path: str = "logs/metrics.csv",
        export_interval: float = 10.0
    ):
        """
        Initialize Stats Collector.

        Args:
            predictor: EWMA predictor for load forecasting
            polling_interval: Interval between stats requests (seconds)
            export_path: Path for CSV metric export
            export_interval: Interval for metric export (seconds)
        """
        self.predictor = predictor
        self.polling_interval = polling_interval
        self.export_path = export_path
        self.export_interval = export_interval

        # Previous stats for delta calculations
        self._prev_port_stats: Dict[Tuple[int, int], PortStats] = {}
        self._prev_flow_stats: Dict[Tuple[int, str], FlowStats] = {}

        # Current computed metrics
        self._link_metrics: Dict[Tuple[int, int], LinkMetrics] = {}

        # Link capacities: (dpid, port_no) -> capacity_mbps
        self._link_capacities: Dict[Tuple[int, int], float] = {}

        # Snapshots history
        self._snapshots: List[NetworkSnapshot] = []
        self._max_snapshots = 1000

        # ECMP baseline tracking
        self._ecmp_baseline_energy: float = 0.0

        # Callbacks
        self._get_energy_callback: Optional[Callable] = None

        # Export tracking
        self._last_export_time = 0.0
        self._csv_initialized = False

        logger.info(
            "stats_collector_initialized",
            polling_interval=polling_interval,
            export_path=export_path
        )

    def set_link_capacity(self, dpid: int, port_no: int, capacity_mbps: float):
        """Set the capacity for a link."""
        self._link_capacities[(dpid, port_no)] = capacity_mbps

    def set_energy_callback(self, callback: Callable):
        """Set callback to get current energy stats."""
        self._get_energy_callback = callback

    def set_ecmp_baseline(self, total_power: float):
        """Set ECMP baseline energy for comparison."""
        self._ecmp_baseline_energy = total_power

    def process_port_stats(
        self,
        dpid: int,
        port_stats: List[PortStats]
    ) -> Dict[int, LinkMetrics]:
        """
        Process port statistics from OFPPortStatsReply.

        Args:
            dpid: Datapath ID of the switch
            port_stats: List of port statistics

        Returns:
            Dict mapping port_no to computed LinkMetrics
        """
        metrics = {}
        current_time = time.time()

        for stats in port_stats:
            port_no = stats.port_no
            key = (dpid, port_no)

            # Skip local and controller ports
            if port_no > 65000:  # OpenFlow reserved ports
                continue

            prev_stats = self._prev_port_stats.get(key)

            if prev_stats:
                # Calculate deltas
                time_delta = stats.timestamp - prev_stats.timestamp
                if time_delta > 0:
                    bytes_delta = (
                        (stats.rx_bytes + stats.tx_bytes) -
                        (prev_stats.rx_bytes + prev_stats.tx_bytes)
                    )
                    packets_delta = (
                        (stats.rx_packets + stats.tx_packets) -
                        (prev_stats.rx_packets + prev_stats.tx_packets)
                    )
                    dropped_delta = (
                        (stats.rx_dropped + stats.tx_dropped) -
                        (prev_stats.rx_dropped + prev_stats.tx_dropped)
                    )
                    errors_delta = (
                        (stats.rx_errors + stats.tx_errors) -
                        (prev_stats.rx_errors + prev_stats.tx_errors)
                    )

                    # Calculate throughput (Mbps)
                    throughput_mbps = (bytes_delta * 8) / (time_delta * 1_000_000)

                    # Calculate utilization
                    capacity = self._link_capacities.get(key, 1000.0)  # Default 1 Gbps
                    utilization = min(100.0, (throughput_mbps / capacity) * 100)

                    # Calculate packet loss
                    total_packets = packets_delta + dropped_delta
                    if total_packets > 0:
                        packet_loss = (dropped_delta / total_packets) * 100
                    else:
                        packet_loss = 0.0

                    # Calculate error rate
                    if packets_delta > 0:
                        error_rate = (errors_delta / packets_delta) * 100
                    else:
                        error_rate = 0.0

                    link_metrics = LinkMetrics(
                        dpid=dpid,
                        port_no=port_no,
                        throughput_mbps=throughput_mbps,
                        utilization_percent=utilization,
                        packet_loss_percent=packet_loss,
                        error_rate=error_rate,
                        timestamp=current_time
                    )

                    metrics[port_no] = link_metrics
                    self._link_metrics[key] = link_metrics

                    # Update EWMA predictor
                    link_stats = LinkStats(
                        timestamp=current_time,
                        bytes_rx=stats.rx_bytes,
                        bytes_tx=stats.tx_bytes,
                        packets_rx=stats.rx_packets,
                        packets_tx=stats.tx_packets,
                        errors=stats.rx_errors + stats.tx_errors,
                        utilization=utilization
                    )
                    self.predictor.update(dpid, port_no, link_stats, capacity)

                    logger.debug(
                        "port_metrics_updated",
                        dpid=dpid,
                        port_no=port_no,
                        throughput=round(throughput_mbps, 2),
                        utilization=round(utilization, 2),
                        packet_loss=round(packet_loss, 3)
                    )

            # Store for next iteration
            self._prev_port_stats[key] = stats

        return metrics

    def process_flow_stats(
        self,
        dpid: int,
        flow_stats: List[FlowStats]
    ) -> Dict:
        """
        Process flow statistics from OFPFlowStatsReply.

        Args:
            dpid: Datapath ID
            flow_stats: List of flow statistics

        Returns:
            Dict with aggregated flow metrics
        """
        current_time = time.time()
        total_bytes = 0
        total_packets = 0
        active_flows = 0

        for stats in flow_stats:
            # Create unique key for flow
            match_key = str(sorted(stats.match.items()))
            key = (dpid, match_key)

            total_bytes += stats.byte_count
            total_packets += stats.packet_count

            prev_stats = self._prev_flow_stats.get(key)
            if prev_stats:
                if stats.byte_count > prev_stats.byte_count:
                    active_flows += 1

            self._prev_flow_stats[key] = stats

        logger.debug(
            "flow_stats_processed",
            dpid=dpid,
            total_flows=len(flow_stats),
            active_flows=active_flows,
            total_bytes=total_bytes
        )

        return {
            "total_flows": len(flow_stats),
            "active_flows": active_flows,
            "total_bytes": total_bytes,
            "total_packets": total_packets,
            "timestamp": current_time
        }

    def get_link_metrics(
        self,
        dpid: int,
        port_no: int
    ) -> Optional[LinkMetrics]:
        """Get latest metrics for a specific link."""
        return self._link_metrics.get((dpid, port_no))

    def get_all_link_metrics(self) -> Dict[Tuple[int, int], LinkMetrics]:
        """Get all link metrics."""
        return dict(self._link_metrics)

    def take_snapshot(self) -> NetworkSnapshot:
        """
        Take a snapshot of network-wide metrics.

        Returns:
            NetworkSnapshot with aggregated metrics
        """
        current_time = time.time()

        # Aggregate link metrics
        active_ports = 0
        sleeping_ports = 0
        total_utilization = 0.0
        max_utilization = 0.0

        for (dpid, port_no), metrics in self._link_metrics.items():
            total_utilization += metrics.utilization_percent
            max_utilization = max(max_utilization, metrics.utilization_percent)
            active_ports += 1

        avg_utilization = (
            total_utilization / active_ports if active_ports > 0 else 0.0
        )

        # Get energy stats
        total_energy = 0.0
        energy_savings = 0.0

        if self._get_energy_callback:
            energy_stats = self._get_energy_callback()
            total_energy = energy_stats.get("total_power_watts", 0.0)
            sleeping_ports = energy_stats.get("sleeping_ports", 0)
            if self._ecmp_baseline_energy > 0:
                energy_savings = (
                    (self._ecmp_baseline_energy - total_energy) /
                    self._ecmp_baseline_energy * 100
                )

        # Aggregate flow stats
        total_flows = len(self._prev_flow_stats)
        total_bytes = sum(
            s.byte_count for s in self._prev_flow_stats.values()
        )
        total_packets = sum(
            s.packet_count for s in self._prev_flow_stats.values()
        )

        snapshot = NetworkSnapshot(
            timestamp=current_time,
            total_flows=total_flows,
            total_bytes=total_bytes,
            total_packets=total_packets,
            active_ports=active_ports,
            sleeping_ports=sleeping_ports,
            average_utilization=avg_utilization,
            max_utilization=max_utilization,
            total_energy_watts=total_energy,
            energy_savings_percent=energy_savings
        )

        # Store snapshot
        if len(self._snapshots) >= self._max_snapshots:
            self._snapshots.pop(0)
        self._snapshots.append(snapshot)

        logger.info(
            "network_snapshot_taken",
            active_ports=active_ports,
            sleeping_ports=sleeping_ports,
            avg_utilization=round(avg_utilization, 2),
            energy_savings=round(energy_savings, 2)
        )

        return snapshot

    def export_metrics(self, force: bool = False):
        """
        Export metrics to CSV file.

        Args:
            force: Force export regardless of interval
        """
        current_time = time.time()

        if not force:
            if current_time - self._last_export_time < self.export_interval:
                return

        self._last_export_time = current_time

        # Ensure directory exists
        export_dir = Path(self.export_path).parent
        export_dir.mkdir(parents=True, exist_ok=True)

        # Take snapshot
        snapshot = self.take_snapshot()

        # Initialize CSV if needed
        if not self._csv_initialized:
            self._initialize_csv()

        # Append metrics
        try:
            with open(self.export_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    snapshot.timestamp,
                    snapshot.total_flows,
                    snapshot.total_bytes,
                    snapshot.total_packets,
                    snapshot.active_ports,
                    snapshot.sleeping_ports,
                    round(snapshot.average_utilization, 2),
                    round(snapshot.max_utilization, 2),
                    round(snapshot.total_energy_watts, 2),
                    round(snapshot.energy_savings_percent, 2),
                    round(snapshot.active_ports / (snapshot.active_ports + snapshot.sleeping_ports) * 100, 2)
                    if (snapshot.active_ports + snapshot.sleeping_ports) > 0 else 0
                ])

            logger.debug(
                "metrics_exported",
                path=self.export_path,
                timestamp=snapshot.timestamp
            )

        except Exception as e:
            logger.error(
                "metrics_export_failed",
                path=self.export_path,
                error=str(e)
            )

    def _initialize_csv(self):
        """Initialize CSV file with headers."""
        try:
            with open(self.export_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "total_flows",
                    "total_bytes",
                    "total_packets",
                    "active_ports",
                    "sleeping_ports",
                    "avg_utilization_percent",
                    "max_utilization_percent",
                    "total_energy_watts",
                    "energy_savings_percent",
                    "active_ports_ratio_percent"
                ])
            self._csv_initialized = True
            logger.info("csv_initialized", path=self.export_path)
        except Exception as e:
            logger.error("csv_init_failed", error=str(e))

    def get_recent_snapshots(self, limit: int = 100) -> List[NetworkSnapshot]:
        """Get recent network snapshots."""
        return self._snapshots[-limit:]

    def get_average_metrics(self, window_seconds: float = 60.0) -> Dict:
        """
        Get average metrics over a time window.

        Args:
            window_seconds: Time window for averaging

        Returns:
            Dict with averaged metrics
        """
        current_time = time.time()
        cutoff_time = current_time - window_seconds

        recent = [s for s in self._snapshots if s.timestamp > cutoff_time]

        if not recent:
            return {
                "average_utilization": 0.0,
                "average_energy": 0.0,
                "average_savings": 0.0,
                "sample_count": 0
            }

        return {
            "average_utilization": sum(
                s.average_utilization for s in recent
            ) / len(recent),
            "average_energy": sum(
                s.total_energy_watts for s in recent
            ) / len(recent),
            "average_savings": sum(
                s.energy_savings_percent for s in recent
            ) / len(recent),
            "sample_count": len(recent)
        }

    def get_ecmp_comparison(self) -> Dict:
        """
        Get comparison with ECMP baseline.

        Returns:
            Dict with comparison metrics
        """
        if not self._snapshots:
            return {
                "baseline_energy": self._ecmp_baseline_energy,
                "current_energy": 0.0,
                "energy_savings_percent": 0.0,
                "active_ports_reduction_percent": 0.0
            }

        latest = self._snapshots[-1]

        return {
            "baseline_energy": self._ecmp_baseline_energy,
            "current_energy": latest.total_energy_watts,
            "energy_savings_percent": latest.energy_savings_percent,
            "energy_savings_watts": max(
                0, self._ecmp_baseline_energy - latest.total_energy_watts
            ),
            "timestamp": latest.timestamp
        }

    def get_qos_metrics(self) -> Dict:
        """Get QoS-related metrics."""
        if not self._link_metrics:
            return {
                "max_packet_loss": 0.0,
                "avg_packet_loss": 0.0,
                "max_latency_estimate_ms": 0.0,
                "qos_violations": 0
            }

        packet_losses = [
            m.packet_loss_percent for m in self._link_metrics.values()
        ]
        utilizations = [
            m.utilization_percent for m in self._link_metrics.values()
        ]

        # Estimate latency increase based on utilization
        # Higher utilization -> higher queuing delay
        max_util = max(utilizations) if utilizations else 0
        latency_estimate = max_util * 0.1  # Simple model: 0.1ms per % utilization

        qos_violations = sum(1 for u in utilizations if u > 80)

        return {
            "max_packet_loss": max(packet_losses) if packet_losses else 0.0,
            "avg_packet_loss": sum(packet_losses) / len(packet_losses) if packet_losses else 0.0,
            "max_latency_estimate_ms": latency_estimate,
            "max_utilization": max_util,
            "qos_violations": qos_violations
        }

    def get_stats(self) -> Dict:
        """Get collector statistics."""
        return {
            "tracked_ports": len(self._link_metrics),
            "tracked_flows": len(self._prev_flow_stats),
            "snapshots_stored": len(self._snapshots),
            "polling_interval": self.polling_interval,
            "export_path": self.export_path,
            "last_export": self._last_export_time
        }

    def reset(self):
        """Reset all collected stats."""
        self._prev_port_stats.clear()
        self._prev_flow_stats.clear()
        self._link_metrics.clear()
        self._snapshots.clear()
        self._csv_initialized = False
        self._last_export_time = 0.0
        logger.info("stats_collector_reset")
