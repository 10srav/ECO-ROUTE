"""
Energy Model Module for EcoRoute SDN Controller

Models switch and port power consumption for energy-aware routing decisions.
Power model: Total_Power = Switch_Base_Power + (Active_Ports * Port_Power)
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import structlog

logger = structlog.get_logger(__name__)


class PortState(Enum):
    """Port operational states."""
    ACTIVE = "active"
    SLEEPING = "sleeping"
    TRANSITIONING = "transitioning"
    DISABLED = "disabled"


@dataclass
class PortPowerState:
    """Power state for a single port."""
    port_no: int
    state: PortState = PortState.ACTIVE
    power_consumption: float = 5.0  # Watts
    last_state_change: float = field(default_factory=time.time)
    wake_up_time: Optional[float] = None  # Time when port will be fully awake


@dataclass
class SwitchPowerState:
    """Power state for a switch with all its ports."""
    dpid: int
    base_power: float = 50.0  # Watts
    ports: Dict[int, PortPowerState] = field(default_factory=dict)
    total_power: float = 0.0

    def calculate_total_power(self) -> float:
        """Calculate total power consumption for this switch."""
        port_power = sum(
            p.power_consumption
            for p in self.ports.values()
            if p.state == PortState.ACTIVE
        )
        # Add minimal power for sleeping ports
        sleep_power = sum(
            0.5  # Sleep mode power
            for p in self.ports.values()
            if p.state == PortState.SLEEPING
        )
        self.total_power = self.base_power + port_power + sleep_power
        return self.total_power


@dataclass
class EnergySnapshot:
    """Snapshot of network energy consumption."""
    timestamp: float
    total_switches: int
    total_ports: int
    active_ports: int
    sleeping_ports: int
    total_power: float  # Watts
    baseline_power: float  # Power if all ports active
    energy_savings_percent: float


class EnergyModel:
    """
    Energy consumption model for SDN data center network.

    Tracks power consumption per switch and port, calculates energy savings
    compared to always-on baseline, and provides cost metrics for routing decisions.
    """

    def __init__(
        self,
        switch_base_power: float = 50.0,
        port_power: float = 5.0,
        sleep_power: float = 0.5,
        wake_latency_ms: float = 100.0
    ):
        """
        Initialize Energy Model.

        Args:
            switch_base_power: Base power per switch (Watts)
            port_power: Power per active port (Watts)
            sleep_power: Power per sleeping port (Watts)
            wake_latency_ms: Time to wake a sleeping port (milliseconds)
        """
        self.switch_base_power = switch_base_power
        self.port_power = port_power
        self.sleep_power = sleep_power
        self.wake_latency_ms = wake_latency_ms

        # Track switches and their power states
        self._switches: Dict[int, SwitchPowerState] = {}

        # Energy metrics history
        self._snapshots: List[EnergySnapshot] = []
        self._max_snapshots = 1000

        # Sleep/wake event log
        self._events: List[Dict] = []
        self._max_events = 500

        logger.info(
            "energy_model_initialized",
            switch_base_power=switch_base_power,
            port_power=port_power,
            sleep_power=sleep_power,
            wake_latency_ms=wake_latency_ms
        )

    def register_switch(self, dpid: int, ports: List[int]) -> SwitchPowerState:
        """
        Register a switch and its ports in the energy model.

        Args:
            dpid: Datapath ID
            ports: List of port numbers

        Returns:
            SwitchPowerState for the registered switch
        """
        switch = SwitchPowerState(
            dpid=dpid,
            base_power=self.switch_base_power,
            ports={
                port_no: PortPowerState(
                    port_no=port_no,
                    power_consumption=self.port_power
                )
                for port_no in ports
            }
        )
        switch.calculate_total_power()
        self._switches[dpid] = switch

        logger.info(
            "switch_registered",
            dpid=dpid,
            num_ports=len(ports),
            total_power=switch.total_power
        )

        return switch

    def unregister_switch(self, dpid: int):
        """Remove a switch from the energy model."""
        if dpid in self._switches:
            del self._switches[dpid]
            logger.info("switch_unregistered", dpid=dpid)

    def set_port_sleeping(self, dpid: int, port_no: int) -> bool:
        """
        Set a port to sleeping state.

        Args:
            dpid: Datapath ID
            port_no: Port number

        Returns:
            True if successful, False otherwise
        """
        switch = self._switches.get(dpid)
        if not switch or port_no not in switch.ports:
            logger.warning(
                "port_not_found",
                dpid=dpid,
                port_no=port_no,
                action="set_sleeping"
            )
            return False

        port = switch.ports[port_no]
        if port.state == PortState.SLEEPING:
            return True  # Already sleeping

        old_state = port.state
        port.state = PortState.SLEEPING
        port.power_consumption = self.sleep_power
        port.last_state_change = time.time()
        port.wake_up_time = None

        switch.calculate_total_power()

        self._log_event("port_sleep", dpid, port_no, old_state, PortState.SLEEPING)

        logger.info(
            "port_set_sleeping",
            dpid=dpid,
            port_no=port_no,
            new_switch_power=switch.total_power
        )

        return True

    def set_port_active(self, dpid: int, port_no: int) -> bool:
        """
        Set a port to active state (wake up).

        Args:
            dpid: Datapath ID
            port_no: Port number

        Returns:
            True if successful, False otherwise
        """
        switch = self._switches.get(dpid)
        if not switch or port_no not in switch.ports:
            logger.warning(
                "port_not_found",
                dpid=dpid,
                port_no=port_no,
                action="set_active"
            )
            return False

        port = switch.ports[port_no]
        if port.state == PortState.ACTIVE:
            return True  # Already active

        old_state = port.state
        port.state = PortState.ACTIVE
        port.power_consumption = self.port_power
        port.last_state_change = time.time()
        port.wake_up_time = time.time() + (self.wake_latency_ms / 1000.0)

        switch.calculate_total_power()

        self._log_event("port_wake", dpid, port_no, old_state, PortState.ACTIVE)

        logger.info(
            "port_set_active",
            dpid=dpid,
            port_no=port_no,
            new_switch_power=switch.total_power,
            wake_latency_ms=self.wake_latency_ms
        )

        return True

    def set_port_transitioning(self, dpid: int, port_no: int) -> bool:
        """Set a port to transitioning state during sleep/wake."""
        switch = self._switches.get(dpid)
        if not switch or port_no not in switch.ports:
            return False

        port = switch.ports[port_no]
        port.state = PortState.TRANSITIONING
        port.last_state_change = time.time()

        return True

    def get_port_state(self, dpid: int, port_no: int) -> Optional[PortState]:
        """Get the current state of a port."""
        switch = self._switches.get(dpid)
        if not switch or port_no not in switch.ports:
            return None
        return switch.ports[port_no].state

    def is_port_active(self, dpid: int, port_no: int) -> bool:
        """Check if a port is active."""
        return self.get_port_state(dpid, port_no) == PortState.ACTIVE

    def is_port_sleeping(self, dpid: int, port_no: int) -> bool:
        """Check if a port is sleeping."""
        return self.get_port_state(dpid, port_no) == PortState.SLEEPING

    def is_port_ready(self, dpid: int, port_no: int) -> bool:
        """Check if a port is ready (active and past wake-up time)."""
        switch = self._switches.get(dpid)
        if not switch or port_no not in switch.ports:
            return False

        port = switch.ports[port_no]
        if port.state != PortState.ACTIVE:
            return False

        if port.wake_up_time and time.time() < port.wake_up_time:
            return False

        return True

    def get_link_energy_cost(
        self,
        src_dpid: int,
        src_port: int,
        dst_dpid: int,
        dst_port: int
    ) -> float:
        """
        Calculate energy cost for using a link.

        Returns higher cost for links requiring wake-up from sleep.

        Args:
            src_dpid: Source switch dpid
            src_port: Source port number
            dst_dpid: Destination switch dpid
            dst_port: Destination port number

        Returns:
            Energy cost (0-1 scale, lower is better)
        """
        cost = 0.0

        # Check source port
        src_state = self.get_port_state(src_dpid, src_port)
        if src_state == PortState.SLEEPING:
            cost += 0.5  # Wake-up penalty
        elif src_state == PortState.TRANSITIONING:
            cost += 0.3  # Transition penalty
        elif src_state == PortState.ACTIVE:
            cost += 0.0  # Already active - lowest cost

        # Check destination port
        dst_state = self.get_port_state(dst_dpid, dst_port)
        if dst_state == PortState.SLEEPING:
            cost += 0.5
        elif dst_state == PortState.TRANSITIONING:
            cost += 0.3
        elif dst_state == PortState.ACTIVE:
            cost += 0.0

        return min(1.0, cost)

    def get_path_energy_cost(
        self,
        path: List[Tuple[int, int, int, int]]  # [(src_dpid, src_port, dst_dpid, dst_port), ...]
    ) -> float:
        """
        Calculate total energy cost for a path.

        Args:
            path: List of link tuples

        Returns:
            Total energy cost for the path
        """
        total_cost = 0.0
        for src_dpid, src_port, dst_dpid, dst_port in path:
            total_cost += self.get_link_energy_cost(
                src_dpid, src_port, dst_dpid, dst_port
            )
        return total_cost

    def calculate_snapshot(self) -> EnergySnapshot:
        """
        Calculate current energy consumption snapshot.

        Returns:
            EnergySnapshot with current metrics
        """
        total_switches = len(self._switches)
        total_ports = 0
        active_ports = 0
        sleeping_ports = 0
        total_power = 0.0
        baseline_power = 0.0

        for switch in self._switches.values():
            total_power += switch.calculate_total_power()
            baseline_power += switch.base_power

            for port in switch.ports.values():
                total_ports += 1
                baseline_power += self.port_power  # All ports active baseline

                if port.state == PortState.ACTIVE:
                    active_ports += 1
                elif port.state == PortState.SLEEPING:
                    sleeping_ports += 1

        # Calculate savings percentage
        if baseline_power > 0:
            energy_savings_percent = (
                (baseline_power - total_power) / baseline_power
            ) * 100
        else:
            energy_savings_percent = 0.0

        snapshot = EnergySnapshot(
            timestamp=time.time(),
            total_switches=total_switches,
            total_ports=total_ports,
            active_ports=active_ports,
            sleeping_ports=sleeping_ports,
            total_power=total_power,
            baseline_power=baseline_power,
            energy_savings_percent=energy_savings_percent
        )

        # Store snapshot
        if len(self._snapshots) >= self._max_snapshots:
            self._snapshots.pop(0)
        self._snapshots.append(snapshot)

        return snapshot

    def get_active_ports_ratio(self) -> float:
        """Calculate ratio of active ports to total ports."""
        total_ports = 0
        active_ports = 0

        for switch in self._switches.values():
            for port in switch.ports.values():
                total_ports += 1
                if port.state == PortState.ACTIVE:
                    active_ports += 1

        if total_ports == 0:
            return 1.0

        return active_ports / total_ports

    def get_sleeping_links(self) -> Set[Tuple[int, int]]:
        """Get all currently sleeping links (dpid, port_no)."""
        sleeping = set()
        for dpid, switch in self._switches.items():
            for port_no, port in switch.ports.items():
                if port.state == PortState.SLEEPING:
                    sleeping.add((dpid, port_no))
        return sleeping

    def get_active_links(self) -> Set[Tuple[int, int]]:
        """Get all currently active links (dpid, port_no)."""
        active = set()
        for dpid, switch in self._switches.items():
            for port_no, port in switch.ports.items():
                if port.state == PortState.ACTIVE:
                    active.add((dpid, port_no))
        return active

    def _log_event(
        self,
        event_type: str,
        dpid: int,
        port_no: int,
        old_state: PortState,
        new_state: PortState
    ):
        """Log a sleep/wake event."""
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "dpid": dpid,
            "port_no": port_no,
            "old_state": old_state.value,
            "new_state": new_state.value
        }

        if len(self._events) >= self._max_events:
            self._events.pop(0)
        self._events.append(event)

    def get_events(self, limit: int = 100) -> List[Dict]:
        """Get recent sleep/wake events."""
        return self._events[-limit:]

    def get_snapshots(self, limit: int = 100) -> List[EnergySnapshot]:
        """Get recent energy snapshots."""
        return self._snapshots[-limit:]

    def get_stats(self) -> Dict:
        """Get comprehensive energy statistics."""
        snapshot = self.calculate_snapshot()

        return {
            "total_switches": snapshot.total_switches,
            "total_ports": snapshot.total_ports,
            "active_ports": snapshot.active_ports,
            "sleeping_ports": snapshot.sleeping_ports,
            "active_ports_ratio": round(self.get_active_ports_ratio(), 3),
            "total_power_watts": round(snapshot.total_power, 2),
            "baseline_power_watts": round(snapshot.baseline_power, 2),
            "power_saved_watts": round(
                snapshot.baseline_power - snapshot.total_power, 2
            ),
            "energy_savings_percent": round(snapshot.energy_savings_percent, 2),
            "recent_events": len(self._events),
            "timestamp": snapshot.timestamp
        }

    def get_switch_stats(self, dpid: int) -> Optional[Dict]:
        """Get power stats for a specific switch."""
        switch = self._switches.get(dpid)
        if not switch:
            return None

        active = sum(
            1 for p in switch.ports.values()
            if p.state == PortState.ACTIVE
        )
        sleeping = sum(
            1 for p in switch.ports.values()
            if p.state == PortState.SLEEPING
        )

        return {
            "dpid": dpid,
            "base_power": switch.base_power,
            "total_power": switch.calculate_total_power(),
            "total_ports": len(switch.ports),
            "active_ports": active,
            "sleeping_ports": sleeping,
            "ports": {
                port_no: {
                    "state": port.state.value,
                    "power": port.power_consumption,
                    "last_change": port.last_state_change
                }
                for port_no, port in switch.ports.items()
            }
        }

    def reset(self):
        """Reset all energy tracking state."""
        self._switches.clear()
        self._snapshots.clear()
        self._events.clear()
        logger.info("energy_model_reset")
