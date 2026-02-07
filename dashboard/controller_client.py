#!/usr/bin/env python3
"""
Controller Client for EcoRoute Dashboard

Provides real-time data from the Ryu controller instead of mock data.
Connects to the controller's REST API (WSGI on port 8080).
Falls back to local simulation when controller is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from controller.ewma_predictor import AdaptiveEWMAPredictor
from controller.energy_model import EnergyModel
from controller.energy_router import EnergyAwareRouter


@dataclass
class ControllerConfig:
    """Configuration for controller connection."""
    host: str = "127.0.0.1"
    port: int = 8080
    use_mock: bool = False
    poll_interval: float = 2.0
    trained_model_path: Optional[str] = None


class ControllerClient:
    """
    Client for connecting to EcoRoute Ryu Controller.

    Provides methods to fetch real-time data about:
    - Network topology
    - Energy consumption
    - Traffic predictions
    - Sleep/wake events

    When the controller is unavailable, falls back to local simulation
    with a proper fat-tree topology.
    """

    def __init__(self, config: Optional[ControllerConfig] = None):
        self.config = config or ControllerConfig()
        self.base_url = f"http://{self.config.host}:{self.config.port}"

        # Local simulation components (used when controller unavailable)
        self._init_local_components()

        # Cache for data
        self._cache: Dict[str, Any] = {}
        self._cache_time: Dict[str, float] = {}
        self._cache_ttl = 1.0  # seconds

        # Connection status
        self.connected = False
        self._last_check = 0

        # Energy history for charts
        self.energy_history: List[Dict] = []

        # Start background polling if using real controller
        self._polling = False
        if not self.config.use_mock:
            self._start_polling()

    def _init_local_components(self):
        """Initialize local EcoRoute components for simulation."""
        model_params = self._load_trained_model()

        self.predictor = AdaptiveEWMAPredictor(
            base_alpha=model_params.get("optimal_alpha", 0.3),
            min_alpha=model_params.get("optimal_min_alpha", 0.1),
            max_alpha=model_params.get("optimal_max_alpha", 0.7),
            prediction_steps=model_params.get("prediction_steps", 3)
        )

        self.energy_model = EnergyModel(
            switch_base_power=50.0,
            port_power=5.0,
            sleep_power=0.5
        )

        self.router = EnergyAwareRouter(
            energy_model=self.energy_model,
            predictor=self.predictor,
            k_paths=3
        )

        # Setup proper fat-tree topology
        self._setup_fat_tree(k=4)

        # Events history
        self._events: List[Dict] = []

        # Track simulation time
        self._sim_start = time.time()

    def _load_trained_model(self) -> Dict:
        """Load trained model parameters."""
        if self.config.trained_model_path:
            path = self.config.trained_model_path
        else:
            path = "training/models/ewma_model.json"

        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to load trained model from %s: %s", path, e)

        return {}

    def _setup_fat_tree(self, k: int = 4):
        """Setup proper fat-tree topology in local components.

        For k=4:
        - Core switches: dpid 1-4
        - Aggregation switches: dpid 5-12 (4 pods x 2 per pod)
        - Edge switches: dpid 13-20 (4 pods x 2 per pod)
        """
        num_core = (k // 2) ** 2          # 4
        num_agg_per_pod = k // 2           # 2
        num_edge_per_pod = k // 2          # 2
        num_pods = k                       # 4

        # Register all switches with k ports each
        total_switches = num_core + num_pods * (num_agg_per_pod + num_edge_per_pod)
        for sw_id in range(1, total_switches + 1):
            self.energy_model.register_switch(sw_id, list(range(1, k + 1)))

        # Core-to-Aggregation links
        # Each agg_idx connects to a specific set of k/2 core switches
        for pod in range(num_pods):
            for a in range(num_agg_per_pod):
                agg_sw = num_core + pod * num_agg_per_pod + a + 1
                for i in range(k // 2):
                    core_sw = a * (k // 2) + i + 1
                    # Agg uplink port: after edge-facing ports
                    agg_port = num_edge_per_pod + i + 1
                    # Core port: one per pod
                    core_port = pod + 1
                    self.router.add_link(core_sw, core_port, agg_sw, agg_port, 1000.0)
                    self.router.add_link(agg_sw, agg_port, core_sw, core_port, 1000.0)

        # Aggregation-to-Edge links (full mesh within each pod)
        for pod in range(num_pods):
            for a in range(num_agg_per_pod):
                agg_sw = num_core + pod * num_agg_per_pod + a + 1
                for e in range(num_edge_per_pod):
                    edge_sw = num_core + num_pods * num_agg_per_pod + pod * num_edge_per_pod + e + 1
                    agg_port = e + 1  # downlink port on agg
                    edge_port = num_agg_per_pod + a + 1  # uplink port on edge
                    self.router.add_link(agg_sw, agg_port, edge_sw, edge_port, 1000.0)
                    self.router.add_link(edge_sw, edge_port, agg_sw, agg_port, 1000.0)

        # Classify node types
        self.router.classify_fat_tree_nodes(k)

    def _start_polling(self):
        """Start background polling thread."""
        self._polling = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def _poll_loop(self):
        """Background polling loop."""
        while self._polling:
            self._check_connection()
            if self.connected:
                self._fetch_all_data()
            time.sleep(self.config.poll_interval)

    def _check_connection(self):
        """Check if controller is reachable."""
        if time.time() - self._last_check < 5:
            return

        try:
            resp = requests.get(f"{self.base_url}/stats", timeout=2)
            self.connected = resp.status_code == 200
        except Exception as e:
            logger.debug("Controller connection check failed: %s", e)
            self.connected = False

        self._last_check = time.time()

    def _fetch_all_data(self):
        """Fetch all data from controller REST API."""
        try:
            resp = requests.get(f"{self.base_url}/topology", timeout=5)
            if resp.ok:
                self._cache["topology"] = resp.json()
                self._cache_time["topology"] = time.time()

            resp = requests.get(f"{self.base_url}/stats", timeout=5)
            if resp.ok:
                self._cache["stats"] = resp.json()
                self._cache_time["stats"] = time.time()

            resp = requests.get(f"{self.base_url}/predictions", timeout=5)
            if resp.ok:
                self._cache["predictions"] = resp.json()
                self._cache_time["predictions"] = time.time()

            resp = requests.get(f"{self.base_url}/energy", timeout=5)
            if resp.ok:
                energy_data = resp.json()
                self._cache["energy"] = energy_data
                self._cache_time["energy"] = time.time()
                # Track energy history for charts
                self.energy_history.append({
                    "timestamp": time.time(),
                    "savings": energy_data.get("energy_savings_percent", 0),
                    "power": energy_data.get("total_power_watts", 0)
                })
                if len(self.energy_history) > 100:
                    self.energy_history.pop(0)

            resp = requests.get(f"{self.base_url}/qos", timeout=5)
            if resp.ok:
                self._cache["qos"] = resp.json()
                self._cache_time["qos"] = time.time()

            resp = requests.get(f"{self.base_url}/events", timeout=5)
            if resp.ok:
                self._cache["events"] = resp.json()
                self._cache_time["events"] = time.time()

            resp = requests.get(f"{self.base_url}/ecmp-comparison", timeout=5)
            if resp.ok:
                self._cache["ecmp_comparison"] = resp.json()
                self._cache_time["ecmp_comparison"] = time.time()

        except Exception as e:
            logger.warning("Failed to fetch data from controller: %s", e)

    def _get_cached(self, key: str) -> Optional[Dict]:
        """Get cached data if still valid."""
        if key in self._cache:
            if time.time() - self._cache_time.get(key, 0) < self._cache_ttl:
                return self._cache[key]
        return None

    def _simulate_traffic(self):
        """Simulate traffic for local mode with sleep/wake decisions."""
        import random
        import numpy as np

        elapsed = time.time() - self._sim_start
        hour = (elapsed / 60) % 24

        # Diurnal pattern
        diurnal = 0.3 + 0.7 * np.sin(np.pi * hour / 12) ** 2

        # Update each link
        for dpid in range(1, 21):
            for port in range(1, 5):
                base_load = 20 + 50 * diurnal
                noise = random.gauss(0, 10)
                load = max(5, min(95, base_load + noise))

                from controller.ewma_predictor import LinkStats
                stats = LinkStats(timestamp=time.time(), utilization=load)
                self.predictor.update(dpid, port, stats)

                # Sleep/wake decisions
                if self.predictor.should_sleep(dpid, port, 20.0, 30.0):
                    if not self.energy_model.is_port_sleeping(dpid, port):
                        self.energy_model.set_port_sleeping(dpid, port)
                        self._events.append({
                            "timestamp": time.time(),
                            "type": "port_sleep",
                            "dpid": dpid,
                            "port": port,
                            "details": f"Link {dpid}:{port} put to sleep (load below threshold)"
                        })
                elif self.predictor.should_wake(dpid, port, 60.0):
                    if self.energy_model.is_port_sleeping(dpid, port):
                        self.energy_model.set_port_active(dpid, port)
                        self._events.append({
                            "timestamp": time.time(),
                            "type": "port_wake",
                            "dpid": dpid,
                            "port": port,
                            "details": f"Link {dpid}:{port} woken up (load increasing)"
                        })

        # Track energy history
        energy_stats = self.energy_model.get_stats()
        self.energy_history.append({
            "timestamp": time.time(),
            "savings": energy_stats.get("energy_savings_percent", 0),
            "power": energy_stats.get("total_power_watts", 0)
        })
        if len(self.energy_history) > 100:
            self.energy_history.pop(0)

        # Keep events limited
        if len(self._events) > 100:
            self._events = self._events[-100:]

    def get_topology(self) -> Dict:
        """Get network topology."""
        # Try cache first
        cached = self._get_cached("topology")
        if cached:
            return cached

        # Try controller
        if self.connected:
            try:
                resp = requests.get(f"{self.base_url}/topology", timeout=5)
                if resp.ok:
                    return resp.json()
            except Exception as e:
                logger.debug("Topology fetch from controller failed: %s", e)

        # Fall back to local simulation
        self._simulate_traffic()
        return self.router.get_topology_info()

    def get_energy_stats(self) -> Dict:
        """Get energy consumption statistics."""
        cached = self._get_cached("energy")
        if cached:
            return cached

        if self.connected:
            try:
                resp = requests.get(f"{self.base_url}/energy", timeout=5)
                if resp.ok:
                    return resp.json()
            except Exception as e:
                logger.debug("Energy fetch from controller failed: %s", e)

        # Local simulation
        self._simulate_traffic()
        return self.energy_model.get_stats()

    def get_predictions(self) -> Dict:
        """Get EWMA traffic predictions."""
        cached = self._get_cached("predictions")
        if cached:
            return cached

        if self.connected:
            try:
                resp = requests.get(f"{self.base_url}/predictions", timeout=5)
                if resp.ok:
                    return resp.json()
            except Exception as e:
                logger.debug("Predictions fetch from controller failed: %s", e)

        # Local simulation
        self._simulate_traffic()
        predictions = []
        all_preds = self.predictor.get_all_predictions()

        for link_id, pred in list(all_preds.items())[:20]:
            predictions.append({
                "link": f"link_{link_id[0]}_{link_id[1]}",
                "current_load": round(pred.current_load, 2),
                "predicted_load": round(pred.predicted_load, 2),
                "confidence": round(pred.confidence, 2),
                "trend": pred.trend
            })

        avg_conf = sum(p["confidence"] for p in predictions) / len(predictions) if predictions else 0

        return {
            "predictions": predictions,
            "average_confidence": round(avg_conf, 2),
            "timestamp": time.time()
        }

    def get_qos_metrics(self) -> Dict:
        """Get QoS metrics from real data (not random)."""
        cached = self._get_cached("qos")
        if cached:
            return cached

        if self.connected:
            try:
                resp = requests.get(f"{self.base_url}/qos", timeout=5)
                if resp.ok:
                    return resp.json()
            except Exception as e:
                logger.debug("QoS fetch from controller failed: %s", e)

        # Compute QoS from predictor state (not random!)
        all_preds = self.predictor.get_all_predictions()
        loads = [p.current_load for p in all_preds.values()] if all_preds else [0]
        max_load = max(loads) if loads else 0
        avg_load = sum(loads) / len(loads) if loads else 0

        # Estimate packet loss from energy model transitions
        sleeping = self.energy_model.get_stats().get("sleeping_ports", 0)
        total = self.energy_model.get_stats().get("total_ports", 80)
        # Minimal packet loss during sleep transitions
        transition_loss = min(0.05, sleeping * 0.001)

        return {
            "max_packet_loss": round(transition_loss, 4),
            "avg_packet_loss": round(transition_loss * 0.5, 4),
            "max_latency_ms": round(max_load * 0.05, 2),
            "avg_latency_ms": round(avg_load * 0.03, 2),
            "max_utilization": round(max_load, 1),
            "qos_violations": sum(1 for l in loads if l > 80),
            "throughput_ratio": round(1.0 - transition_loss, 3),
            "timestamp": time.time()
        }

    def get_events(self, limit: int = 50) -> List[Dict]:
        """Get recent sleep/wake events."""
        cached = self._get_cached("events")
        if cached:
            events = cached.get("events", [])
            return events[-limit:]

        return self._events[-limit:]

    def get_ecmp_comparison(self) -> Dict:
        """Get ECMP baseline comparison."""
        cached = self._get_cached("ecmp_comparison")
        if cached:
            return cached

        if self.connected:
            try:
                resp = requests.get(f"{self.base_url}/ecmp-comparison", timeout=5)
                if resp.ok:
                    return resp.json()
            except Exception as e:
                logger.debug("ECMP comparison fetch from controller failed: %s", e)

        stats = self.get_energy_stats()
        baseline = stats.get("baseline_power_watts", 1000)
        current = stats.get("total_power_watts", 800)
        savings = (baseline - current) / baseline * 100 if baseline > 0 else 0

        return {
            "baseline_energy_watts": baseline,
            "current_energy_watts": current,
            "energy_savings_percent": round(savings, 2),
            "energy_savings_watts": round(baseline - current, 2),
            "active_ports_baseline": stats.get("total_ports", 80),
            "active_ports_current": stats.get("active_ports", 60),
            "active_ports_reduction_percent": round(
                (1 - stats.get("active_ports_ratio", 0.8)) * 100, 2
            ),
            "timestamp": time.time()
        }

    def get_all_stats(self) -> Dict:
        """Get all statistics."""
        return {
            "energy": self.get_energy_stats(),
            "predictions": self.get_predictions(),
            "qos": self.get_qos_metrics(),
            "ecmp_comparison": self.get_ecmp_comparison(),
            "connected_to_controller": self.connected,
            "timestamp": time.time()
        }

    def close(self):
        """Clean up resources."""
        self._polling = False
