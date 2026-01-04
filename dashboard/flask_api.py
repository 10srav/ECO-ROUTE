#!/usr/bin/env python3
"""
Flask Dashboard API for EcoRoute SDN Controller

REST API endpoints for the EcoRoute monitoring dashboard:
- Real-time topology and statistics
- Energy metrics and savings
- Sleep/wake event history
- EWMA predictions
- QoS metrics

Production-ready: Uses trained EWMA model and realistic traffic simulation.

Usage:
    python flask_api.py [--port PORT] [--controller-url URL] [--use-mock]

Author: EcoRoute Team
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit

import structlog

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.controller_client import ControllerClient, ControllerConfig

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='frontend/build', static_url_path='')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'ecoroute-secret-key')

# Enable CORS
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize SocketIO for real-time updates
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Controller connection configuration
CONTROLLER_URL = os.environ.get('CONTROLLER_URL', 'http://127.0.0.1:8080')

# Global controller client (initialized in main)
controller_client: Optional[ControllerClient] = None


class MockControllerData:
    """
    Mock data generator for development/demo.
    In production, this would connect to the actual Ryu controller.
    """

    def __init__(self):
        self.start_time = time.time()
        self.k = 4  # Fat-tree parameter

        # Generate topology
        self._generate_topology()

        # Initialize metrics history
        self.energy_history = []
        self.utilization_history = []
        self.events = []

    def _generate_topology(self):
        """Generate a fat-tree topology representation."""
        k = self.k
        num_core = (k // 2) ** 2
        num_pods = k

        self.nodes = []
        self.edges = []

        # Core switches
        for i in range(num_core):
            self.nodes.append({
                "id": f"c{i+1}",
                "type": "core",
                "dpid": i + 1,
                "x": 200 + i * 100,
                "y": 50
            })

        # Pods
        for pod in range(num_pods):
            # Aggregation switches
            for a in range(k // 2):
                node_id = f"a{pod}_{a}"
                self.nodes.append({
                    "id": node_id,
                    "type": "aggregation",
                    "dpid": num_core + pod * (k // 2) + a + 1,
                    "x": pod * 200 + 100 + a * 50,
                    "y": 150
                })

            # Edge switches
            for e in range(k // 2):
                node_id = f"e{pod}_{e}"
                self.nodes.append({
                    "id": node_id,
                    "type": "edge",
                    "dpid": num_core + num_pods * (k // 2) + pod * (k // 2) + e + 1,
                    "x": pod * 200 + 100 + e * 50,
                    "y": 250
                })

                # Hosts
                for h in range(k // 2):
                    host_id = f"h{pod}_{e}_{h}"
                    self.nodes.append({
                        "id": host_id,
                        "type": "host",
                        "ip": f"10.{pod}.{e}.{h+1}",
                        "x": pod * 200 + 75 + e * 50 + h * 20,
                        "y": 350
                    })

        # Generate links
        self._generate_links()

    def _generate_links(self):
        """Generate links between nodes."""
        k = self.k
        num_core = (k // 2) ** 2

        # Core to aggregation
        for pod in range(k):
            for a in range(k // 2):
                for c in range(k // 2):
                    core_idx = a * (k // 2) + c
                    if core_idx < num_core:
                        self.edges.append({
                            "source": f"c{core_idx + 1}",
                            "target": f"a{pod}_{a}",
                            "capacity": 10000,
                            "utilization": random.uniform(0, 30),
                            "sleeping": False
                        })

        # Aggregation to edge
        for pod in range(k):
            for a in range(k // 2):
                for e in range(k // 2):
                    self.edges.append({
                        "source": f"a{pod}_{a}",
                        "target": f"e{pod}_{e}",
                        "capacity": 1000,
                        "utilization": random.uniform(0, 50),
                        "sleeping": False
                    })

        # Edge to hosts
        for pod in range(k):
            for e in range(k // 2):
                for h in range(k // 2):
                    self.edges.append({
                        "source": f"e{pod}_{e}",
                        "target": f"h{pod}_{e}_{h}",
                        "capacity": 1000,
                        "utilization": random.uniform(0, 40),
                        "sleeping": False
                    })

    def get_topology(self) -> Dict:
        """Get current topology state."""
        # Update some random links as sleeping
        sleeping_count = 0
        for edge in self.edges:
            if edge['utilization'] < 15 and random.random() > 0.7:
                edge['sleeping'] = True
                sleeping_count += 1
            else:
                edge['sleeping'] = False
            # Update utilization
            edge['utilization'] = max(0, min(100,
                edge['utilization'] + random.uniform(-5, 5)
            ))

        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "total_nodes": len([n for n in self.nodes if n['type'] != 'host']),
            "total_edges": len(self.edges),
            "sleeping_links": sleeping_count
        }

    def get_energy_stats(self) -> Dict:
        """Get energy statistics."""
        uptime = time.time() - self.start_time

        # Simulate energy savings increasing over time
        base_savings = min(30, uptime / 60)  # Max 30% over 30 minutes
        savings_variation = random.uniform(-2, 2)
        current_savings = max(0, base_savings + savings_variation)

        total_ports = len(self.edges)
        sleeping_ports = sum(1 for e in self.edges if e.get('sleeping', False))
        active_ports = total_ports - sleeping_ports

        # Power calculation
        num_switches = len([n for n in self.nodes if n['type'] != 'host'])
        switch_power = num_switches * 50  # 50W per switch
        active_port_power = active_ports * 5  # 5W per active port
        sleep_port_power = sleeping_ports * 0.5  # 0.5W per sleeping port

        total_power = switch_power + active_port_power + sleep_port_power
        baseline_power = switch_power + total_ports * 5

        stats = {
            "total_switches": num_switches,
            "total_ports": total_ports,
            "active_ports": active_ports,
            "sleeping_ports": sleeping_ports,
            "active_ports_ratio": round(active_ports / total_ports, 3) if total_ports > 0 else 1,
            "total_power_watts": round(total_power, 2),
            "baseline_power_watts": round(baseline_power, 2),
            "power_saved_watts": round(baseline_power - total_power, 2),
            "energy_savings_percent": round(current_savings, 2),
            "timestamp": time.time()
        }

        # Store in history
        self.energy_history.append({
            "timestamp": stats["timestamp"],
            "savings": stats["energy_savings_percent"],
            "power": stats["total_power_watts"]
        })

        # Keep last 100 entries
        if len(self.energy_history) > 100:
            self.energy_history.pop(0)

        return stats

    def get_predictions(self) -> Dict:
        """Get EWMA predictions for links."""
        predictions = []

        for edge in self.edges[:20]:  # Limit for demo
            current = edge['utilization']
            # Simulate EWMA prediction
            predicted = current + random.uniform(-5, 5)
            trend = "stable"
            if predicted > current + 3:
                trend = "increasing"
            elif predicted < current - 3:
                trend = "decreasing"

            predictions.append({
                "link": f"{edge['source']}->{edge['target']}",
                "current_load": round(current, 2),
                "predicted_load": round(max(0, min(100, predicted)), 2),
                "confidence": round(random.uniform(0.7, 0.95), 2),
                "trend": trend
            })

        return {
            "predictions": predictions,
            "average_confidence": round(
                sum(p['confidence'] for p in predictions) / len(predictions), 2
            ) if predictions else 0,
            "timestamp": time.time()
        }

    def get_qos_metrics(self) -> Dict:
        """Get QoS metrics."""
        return {
            "max_packet_loss": round(random.uniform(0, 0.05), 3),
            "avg_packet_loss": round(random.uniform(0, 0.02), 3),
            "max_latency_ms": round(random.uniform(1, 5), 2),
            "avg_latency_ms": round(random.uniform(0.5, 2), 2),
            "max_utilization": max(e['utilization'] for e in self.edges),
            "qos_violations": random.randint(0, 2),
            "throughput_ratio": round(random.uniform(0.95, 0.99), 3),
            "timestamp": time.time()
        }

    def get_events(self, limit: int = 50) -> List[Dict]:
        """Get recent sleep/wake events."""
        # Generate some sample events
        event_types = ["port_sleep", "port_wake", "flow_reroute", "prediction_update"]

        if len(self.events) < 10 or random.random() > 0.7:
            event = {
                "timestamp": time.time(),
                "type": random.choice(event_types),
                "dpid": random.randint(1, 20),
                "port": random.randint(1, 4),
                "details": f"Event on switch {random.randint(1, 20)}"
            }
            self.events.append(event)

            # Keep last 100 events
            if len(self.events) > 100:
                self.events.pop(0)

        return self.events[-limit:]

    def get_ecmp_comparison(self) -> Dict:
        """Get ECMP baseline comparison."""
        energy_stats = self.get_energy_stats()

        return {
            "baseline_energy_watts": energy_stats["baseline_power_watts"],
            "current_energy_watts": energy_stats["total_power_watts"],
            "energy_savings_percent": energy_stats["energy_savings_percent"],
            "energy_savings_watts": energy_stats["power_saved_watts"],
            "active_ports_baseline": energy_stats["total_ports"],
            "active_ports_current": energy_stats["active_ports"],
            "active_ports_reduction_percent": round(
                (1 - energy_stats["active_ports_ratio"]) * 100, 2
            ),
            "timestamp": time.time()
        }


# Initialize data source (will be replaced by controller_client in main())
mock_data = MockControllerData()


def get_data_source():
    """Get the appropriate data source (controller client or mock)."""
    global controller_client
    if controller_client is not None:
        return controller_client
    return mock_data


# --- API Routes ---

@app.route('/')
def index():
    """Serve React frontend."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    })


@app.route('/api/topology')
def get_topology():
    """Get current network topology."""
    try:
        data_source = get_data_source()
        topology = data_source.get_topology()
        return jsonify(topology)
    except Exception as e:
        logger.error("topology_fetch_failed", error=str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/api/stats')
def get_stats():
    """Get comprehensive network statistics."""
    try:
        data_source = get_data_source()
        if hasattr(data_source, 'get_all_stats'):
            # Use ControllerClient's comprehensive stats
            return jsonify(data_source.get_all_stats())
        else:
            # Fallback to mock data format
            return jsonify({
                "energy": data_source.get_energy_stats(),
                "predictions": data_source.get_predictions(),
                "qos": data_source.get_qos_metrics(),
                "ecmp_comparison": data_source.get_ecmp_comparison(),
                "timestamp": time.time()
            })
    except Exception as e:
        logger.error("stats_fetch_failed", error=str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/api/energy')
def get_energy():
    """Get energy statistics."""
    try:
        data_source = get_data_source()
        return jsonify(data_source.get_energy_stats())
    except Exception as e:
        logger.error("energy_fetch_failed", error=str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/api/energy/history')
def get_energy_history():
    """Get energy history for charts."""
    try:
        data_source = get_data_source()
        if hasattr(data_source, 'energy_history'):
            history = data_source.energy_history
        else:
            # ControllerClient doesn't have energy_history, build from current stats
            history = []
        return jsonify({
            "history": history,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error("energy_history_fetch_failed", error=str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/api/predictions')
def get_predictions():
    """Get EWMA predictions."""
    try:
        data_source = get_data_source()
        return jsonify(data_source.get_predictions())
    except Exception as e:
        logger.error("predictions_fetch_failed", error=str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/api/qos')
def get_qos():
    """Get QoS metrics."""
    try:
        data_source = get_data_source()
        return jsonify(data_source.get_qos_metrics())
    except Exception as e:
        logger.error("qos_fetch_failed", error=str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/api/events')
def get_events():
    """Get recent events."""
    try:
        data_source = get_data_source()
        limit = request.args.get('limit', 50, type=int)
        return jsonify({
            "events": data_source.get_events(limit),
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error("events_fetch_failed", error=str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/api/ecmp-comparison')
def get_ecmp_comparison():
    """Get ECMP baseline comparison."""
    try:
        data_source = get_data_source()
        return jsonify(data_source.get_ecmp_comparison())
    except Exception as e:
        logger.error("ecmp_comparison_failed", error=str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/api/switches')
def get_switches():
    """Get switch information."""
    try:
        data_source = get_data_source()
        if hasattr(data_source, 'nodes'):
            switches = [n for n in data_source.nodes if n['type'] != 'host']
        else:
            # For ControllerClient, get topology and extract switches
            topo = data_source.get_topology()
            switches = [n for n in topo.get('nodes', []) if n.get('type') != 'host']
        return jsonify({
            "switches": switches,
            "count": len(switches),
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error("switches_fetch_failed", error=str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/api/hosts')
def get_hosts():
    """Get host information."""
    try:
        data_source = get_data_source()
        if hasattr(data_source, 'nodes'):
            hosts = [n for n in data_source.nodes if n['type'] == 'host']
        else:
            # For ControllerClient, get topology and extract hosts
            topo = data_source.get_topology()
            hosts = [n for n in topo.get('nodes', []) if n.get('type') == 'host']
        return jsonify({
            "hosts": hosts,
            "count": len(hosts),
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error("hosts_fetch_failed", error=str(e))
        return jsonify({"error": str(e)}), 500


# --- WebSocket Events ---

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info("client_connected", sid=request.sid)
    emit('connected', {'status': 'connected', 'timestamp': time.time()})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info("client_disconnected", sid=request.sid)


@socketio.on('subscribe_updates')
def handle_subscribe(data):
    """Handle subscription to real-time updates."""
    logger.info("client_subscribed", sid=request.sid)
    emit('subscribed', {'status': 'subscribed', 'timestamp': time.time()})


def broadcast_updates():
    """Background task to broadcast updates."""
    while True:
        socketio.sleep(1)
        try:
            data_source = get_data_source()

            # Emit energy stats
            socketio.emit('energy_update', data_source.get_energy_stats())

            # Emit topology updates less frequently
            if int(time.time()) % 5 == 0:
                socketio.emit('topology_update', data_source.get_topology())

            # Emit events
            recent_events = data_source.get_events(5)
            if recent_events:
                socketio.emit('event_update', {'events': recent_events})

        except Exception as e:
            logger.error("broadcast_failed", error=str(e))


def main():
    """Main entry point."""
    global controller_client

    parser = argparse.ArgumentParser(description="EcoRoute Dashboard API")
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run on (default: 5000)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--controller-url",
        type=str,
        default="http://127.0.0.1:8080",
        help="Ryu controller REST URL"
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock data instead of real controller client"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="training/models/ewma_model.json",
        help="Path to trained EWMA model"
    )

    args = parser.parse_args()

    global CONTROLLER_URL
    CONTROLLER_URL = args.controller_url

    # Initialize controller client unless using mock data
    if not args.use_mock:
        # Parse controller URL
        from urllib.parse import urlparse
        parsed = urlparse(args.controller_url)

        config = ControllerConfig(
            host=parsed.hostname or "127.0.0.1",
            port=parsed.port or 8080,
            use_mock=False,
            trained_model_path=args.model_path
        )
        controller_client = ControllerClient(config)
        logger.info(
            "using_controller_client",
            model_path=args.model_path,
            controller_url=args.controller_url
        )
    else:
        logger.info("using_mock_data")

    logger.info(
        "starting_dashboard_api",
        host=args.host,
        port=args.port,
        controller_url=CONTROLLER_URL,
        mode="mock" if args.use_mock else "production"
    )

    # Start background update task
    socketio.start_background_task(broadcast_updates)

    # Run server
    socketio.run(
        app,
        host=args.host,
        port=args.port,
        debug=args.debug
    )


if __name__ == '__main__':
    main()
