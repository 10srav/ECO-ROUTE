"""
Sleep Manager Module for EcoRoute SDN Controller

Implements Make-Before-Break (MBB) link sleep/wake logic:
1. Reroute all flows to alternate paths BEFORE sleeping link
2. Validate new paths are carrying traffic
3. Only then put original link to sleep
4. Wake links proactively when predicted load increases
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple

import structlog

from controller.energy_model import EnergyModel, PortState
from controller.energy_router import EnergyAwareRouter, Flow, PathScore
from controller.ewma_predictor import AdaptiveEWMAPredictor

logger = structlog.get_logger(__name__)


class TransitionState(Enum):
    """State of a link during sleep/wake transition."""
    IDLE = "idle"
    PREPARING_SLEEP = "preparing_sleep"
    REROUTING = "rerouting"
    VALIDATING = "validating"
    SLEEPING = "sleeping"
    WAKING = "waking"
    ACTIVE = "active"
    FAILED = "failed"


@dataclass
class LinkTransition:
    """Tracks a link's sleep/wake transition."""
    src_dpid: int
    src_port: int
    dst_dpid: int
    dst_port: int
    state: TransitionState = TransitionState.IDLE
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    flows_rerouted: List[str] = field(default_factory=list)
    reroute_paths: Dict[str, PathScore] = field(default_factory=dict)
    error: Optional[str] = None
    retry_count: int = 0


@dataclass
class WakeRequest:
    """Request to wake a sleeping link."""
    src_dpid: int
    src_port: int
    dst_dpid: int
    dst_port: int
    reason: str
    requested_at: float = field(default_factory=time.time)
    priority: int = 0  # Higher = more urgent


class SleepManager:
    """
    Make-Before-Break Sleep Manager

    Coordinates safe link sleep/wake transitions:
    - Ensures flows are rerouted before sleeping
    - Validates traffic is flowing on new paths
    - Handles wake-up latency (100ms default)
    - Proactive wake based on predicted load
    - Rollback on failure
    """

    def __init__(
        self,
        energy_model: EnergyModel,
        router: EnergyAwareRouter,
        predictor: AdaptiveEWMAPredictor,
        sleep_threshold: float = 20.0,
        wake_threshold: float = 60.0,
        min_sleep_duration: float = 30.0,
        wake_latency_ms: float = 100.0,
        max_packet_loss: float = 0.1,
        validation_timeout: float = 5.0,
        max_retries: int = 3
    ):
        """
        Initialize Sleep Manager.

        Args:
            energy_model: Energy tracking model
            router: Energy-aware router
            predictor: EWMA traffic predictor
            sleep_threshold: Utilization below which to sleep (%)
            wake_threshold: Utilization above which to wake (%)
            min_sleep_duration: Min time at low load before sleeping (s)
            wake_latency_ms: Time to wake a port (ms)
            max_packet_loss: Maximum acceptable packet loss during transition (%)
            validation_timeout: Time to validate new paths (s)
            max_retries: Maximum retry attempts for failed transitions
        """
        self.energy_model = energy_model
        self.router = router
        self.predictor = predictor
        self.sleep_threshold = sleep_threshold
        self.wake_threshold = wake_threshold
        self.min_sleep_duration = min_sleep_duration
        self.wake_latency_ms = wake_latency_ms
        self.max_packet_loss = max_packet_loss
        self.validation_timeout = validation_timeout
        self.max_retries = max_retries

        # Active transitions
        self._transitions: Dict[Tuple[int, int], LinkTransition] = {}

        # Wake request queue
        self._wake_queue: List[WakeRequest] = []

        # Callbacks for OpenFlow operations
        self._flow_mod_callback: Optional[Callable] = None
        self._port_mod_callback: Optional[Callable] = None
        self._get_packet_loss_callback: Optional[Callable] = None

        # Statistics
        self._stats = {
            "sleep_attempts": 0,
            "sleep_successes": 0,
            "sleep_failures": 0,
            "wake_attempts": 0,
            "wake_successes": 0,
            "wake_failures": 0,
            "flows_rerouted": 0,
            "rollbacks": 0
        }

        logger.info(
            "sleep_manager_initialized",
            sleep_threshold=sleep_threshold,
            wake_threshold=wake_threshold,
            wake_latency_ms=wake_latency_ms
        )

    def set_flow_mod_callback(self, callback: Callable):
        """Set callback for flow modifications."""
        self._flow_mod_callback = callback

    def set_port_mod_callback(self, callback: Callable):
        """Set callback for port modifications (sleep/wake)."""
        self._port_mod_callback = callback

    def set_packet_loss_callback(self, callback: Callable):
        """Set callback to get current packet loss."""
        self._get_packet_loss_callback = callback

    def get_sleep_candidates(self) -> List[Tuple[int, int, int, int]]:
        """
        Get links that are candidates for sleeping.

        Returns:
            List of (src_dpid, src_port, dst_dpid, dst_port) tuples
        """
        candidates = []

        for (src_dpid, dst_dpid), (src_port, dst_port, _) in self.router._link_info.items():
            # Skip if already in transition
            if (src_dpid, src_port) in self._transitions:
                continue

            # Skip if already sleeping
            if self.energy_model.is_port_sleeping(src_dpid, src_port):
                continue

            # Check EWMA prediction
            if not self.predictor.should_sleep(
                src_dpid, src_port,
                self.sleep_threshold,
                self.min_sleep_duration
            ):
                continue

            # Check if flows can be rerouted
            flows = self.router.get_flows_on_link(
                src_dpid, src_port, dst_dpid, dst_port
            )
            if flows and not self.router.can_reroute_flows(
                flows, src_dpid, dst_dpid
            ):
                continue

            candidates.append((src_dpid, src_port, dst_dpid, dst_port))

        return candidates

    def get_wake_candidates(self) -> List[Tuple[int, int, int, int]]:
        """
        Get sleeping links that should be woken up.

        Returns:
            List of (src_dpid, src_port, dst_dpid, dst_port) tuples
        """
        candidates = []

        for (src_dpid, dst_dpid), (src_port, dst_port, _) in self.router._link_info.items():
            # Only consider sleeping links
            if not self.energy_model.is_port_sleeping(src_dpid, src_port):
                continue

            # Skip if already waking
            key = (src_dpid, src_port)
            if key in self._transitions:
                trans = self._transitions[key]
                if trans.state == TransitionState.WAKING:
                    continue

            # Check EWMA prediction for wake
            if self.predictor.should_wake(
                src_dpid, src_port,
                self.wake_threshold
            ):
                candidates.append((src_dpid, src_port, dst_dpid, dst_port))

        return candidates

    async def initiate_sleep(
        self,
        src_dpid: int,
        src_port: int,
        dst_dpid: int,
        dst_port: int
    ) -> bool:
        """
        Initiate Make-Before-Break sleep transition for a link.

        Steps:
        1. Find alternate paths for all flows on the link
        2. Install new flow rules (make new paths)
        3. Validate traffic is flowing on new paths
        4. Put link to sleep (break old path)

        Args:
            src_dpid: Source switch
            src_port: Source port
            dst_dpid: Destination switch
            dst_port: Destination port

        Returns:
            True if sleep transition successful
        """
        link_key = (src_dpid, src_port)

        # Check if already transitioning
        if link_key in self._transitions:
            logger.warning(
                "sleep_already_in_progress",
                src_dpid=src_dpid,
                src_port=src_port
            )
            return False

        self._stats["sleep_attempts"] += 1

        # Create transition record
        transition = LinkTransition(
            src_dpid=src_dpid,
            src_port=src_port,
            dst_dpid=dst_dpid,
            dst_port=dst_port,
            state=TransitionState.PREPARING_SLEEP
        )
        self._transitions[link_key] = transition

        logger.info(
            "sleep_transition_started",
            src_dpid=src_dpid,
            src_port=src_port,
            dst_dpid=dst_dpid,
            dst_port=dst_port
        )

        try:
            # Step 1: Get flows on this link
            flows = self.router.get_flows_on_link(
                src_dpid, src_port, dst_dpid, dst_port
            )

            if flows:
                # Step 2: Find alternate paths
                transition.state = TransitionState.REROUTING
                excluded = {(src_dpid, dst_dpid)}
                reroute_paths = self.router.find_reroute_paths(flows, excluded)

                if not reroute_paths:
                    raise Exception("No alternate paths found for flows")

                transition.reroute_paths = reroute_paths

                # Step 3: Install new flow rules
                for flow_id, new_path in reroute_paths.items():
                    if self._flow_mod_callback:
                        flow = self.router._flows[flow_id]
                        await self._install_new_path(flow, new_path)
                        transition.flows_rerouted.append(flow_id)
                        self._stats["flows_rerouted"] += 1

                # Step 4: Validate new paths
                transition.state = TransitionState.VALIDATING
                validation_success = await self._validate_paths(transition)

                if not validation_success:
                    raise Exception("Path validation failed")

            # Step 5: Put link to sleep
            transition.state = TransitionState.SLEEPING
            await self._sleep_port(src_dpid, src_port)
            await self._sleep_port(dst_dpid, dst_port)

            # Update energy model
            self.energy_model.set_port_sleeping(src_dpid, src_port)
            self.energy_model.set_port_sleeping(dst_dpid, dst_port)

            # Update flows in router
            for flow_id, new_path in transition.reroute_paths.items():
                old_flow = self.router.remove_flow(flow_id)
                if old_flow:
                    self.router.install_flow(
                        flow_id,
                        old_flow.src_ip,
                        old_flow.dst_ip,
                        new_path,
                        old_flow.bandwidth,
                        old_flow.priority
                    )

            transition.state = TransitionState.SLEEPING
            transition.completed_at = time.time()

            logger.info(
                "sleep_transition_completed",
                src_dpid=src_dpid,
                src_port=src_port,
                flows_rerouted=len(transition.flows_rerouted),
                duration=round(transition.completed_at - transition.started_at, 3)
            )

            self._stats["sleep_successes"] += 1
            return True

        except Exception as e:
            logger.error(
                "sleep_transition_failed",
                src_dpid=src_dpid,
                src_port=src_port,
                error=str(e)
            )

            transition.state = TransitionState.FAILED
            transition.error = str(e)
            self._stats["sleep_failures"] += 1

            # Attempt rollback
            await self._rollback_sleep(transition)

            return False

        finally:
            # Clean up transition after completion or failure
            if link_key in self._transitions:
                trans = self._transitions[link_key]
                if trans.state in [TransitionState.SLEEPING, TransitionState.FAILED]:
                    del self._transitions[link_key]

    async def initiate_wake(
        self,
        src_dpid: int,
        src_port: int,
        dst_dpid: int,
        dst_port: int,
        reason: str = "predicted_load"
    ) -> bool:
        """
        Wake up a sleeping link.

        Args:
            src_dpid: Source switch
            src_port: Source port
            dst_dpid: Destination switch
            dst_port: Destination port
            reason: Reason for waking

        Returns:
            True if wake successful
        """
        link_key = (src_dpid, src_port)

        self._stats["wake_attempts"] += 1

        # Create transition record
        transition = LinkTransition(
            src_dpid=src_dpid,
            src_port=src_port,
            dst_dpid=dst_dpid,
            dst_port=dst_port,
            state=TransitionState.WAKING
        )
        self._transitions[link_key] = transition

        logger.info(
            "wake_transition_started",
            src_dpid=src_dpid,
            src_port=src_port,
            reason=reason
        )

        try:
            # Wake up ports
            await self._wake_port(src_dpid, src_port)
            await self._wake_port(dst_dpid, dst_port)

            # Wait for wake latency
            await asyncio.sleep(self.wake_latency_ms / 1000.0)

            # Validate connectivity
            await self._validate_link_connectivity(
                src_dpid, src_port, dst_dpid, dst_port
            )

            # Update energy model
            self.energy_model.set_port_active(src_dpid, src_port)
            self.energy_model.set_port_active(dst_dpid, dst_port)

            transition.state = TransitionState.ACTIVE
            transition.completed_at = time.time()

            logger.info(
                "wake_transition_completed",
                src_dpid=src_dpid,
                src_port=src_port,
                duration=round(transition.completed_at - transition.started_at, 3)
            )

            self._stats["wake_successes"] += 1
            return True

        except Exception as e:
            logger.error(
                "wake_transition_failed",
                src_dpid=src_dpid,
                src_port=src_port,
                error=str(e)
            )

            transition.state = TransitionState.FAILED
            transition.error = str(e)
            self._stats["wake_failures"] += 1

            return False

        finally:
            if link_key in self._transitions:
                del self._transitions[link_key]

    async def _install_new_path(self, flow: Flow, new_path: PathScore):
        """Install flow rules for new path."""
        if self._flow_mod_callback:
            await self._flow_mod_callback(
                flow.flow_id,
                flow.src_ip,
                flow.dst_ip,
                new_path.path,
                new_path.links
            )

    async def _validate_paths(self, transition: LinkTransition) -> bool:
        """
        Validate that rerouted flows are working.

        Checks packet loss is within acceptable limits.
        """
        # Wait for validation period
        await asyncio.sleep(self.validation_timeout)

        if self._get_packet_loss_callback:
            for flow_id in transition.flows_rerouted:
                packet_loss = await self._get_packet_loss_callback(flow_id)
                if packet_loss > self.max_packet_loss:
                    logger.warning(
                        "path_validation_high_loss",
                        flow_id=flow_id,
                        packet_loss=packet_loss,
                        threshold=self.max_packet_loss
                    )
                    return False

        return True

    async def _sleep_port(self, dpid: int, port: int):
        """Send port sleep command."""
        if self._port_mod_callback:
            await self._port_mod_callback(dpid, port, sleep=True)

    async def _wake_port(self, dpid: int, port: int):
        """Send port wake command."""
        if self._port_mod_callback:
            await self._port_mod_callback(dpid, port, sleep=False)

    async def _validate_link_connectivity(
        self,
        src_dpid: int,
        src_port: int,
        dst_dpid: int,
        dst_port: int
    ):
        """Validate link is operational after wake."""
        # In production, would check link state via OpenFlow
        pass

    async def _rollback_sleep(self, transition: LinkTransition):
        """Rollback a failed sleep transition."""
        self._stats["rollbacks"] += 1

        logger.info(
            "rolling_back_sleep",
            src_dpid=transition.src_dpid,
            src_port=transition.src_port
        )

        # Restore original flows if they were modified
        if transition.flows_rerouted:
            for flow_id in transition.flows_rerouted:
                # Would restore original flow rules here
                pass

        # Ensure port stays active
        self.energy_model.set_port_active(transition.src_dpid, transition.src_port)
        self.energy_model.set_port_active(transition.dst_dpid, transition.dst_port)

    def request_wake(
        self,
        src_dpid: int,
        src_port: int,
        dst_dpid: int,
        dst_port: int,
        reason: str,
        priority: int = 0
    ):
        """Add a wake request to the queue."""
        request = WakeRequest(
            src_dpid=src_dpid,
            src_port=src_port,
            dst_dpid=dst_dpid,
            dst_port=dst_port,
            reason=reason,
            priority=priority
        )
        self._wake_queue.append(request)
        self._wake_queue.sort(key=lambda r: -r.priority)

        logger.debug(
            "wake_request_queued",
            src_dpid=src_dpid,
            src_port=src_port,
            reason=reason
        )

    async def process_wake_queue(self) -> int:
        """
        Process pending wake requests.

        Returns:
            Number of links woken
        """
        woken = 0

        while self._wake_queue:
            request = self._wake_queue.pop(0)

            # Skip if no longer sleeping
            if not self.energy_model.is_port_sleeping(
                request.src_dpid, request.src_port
            ):
                continue

            success = await self.initiate_wake(
                request.src_dpid,
                request.src_port,
                request.dst_dpid,
                request.dst_port,
                request.reason
            )

            if success:
                woken += 1

        return woken

    async def run_optimization_cycle(self) -> Dict:
        """
        Run one cycle of sleep/wake optimization.

        Returns:
            Dict with cycle results
        """
        results = {
            "links_slept": 0,
            "links_woken": 0,
            "flows_rerouted": 0,
            "errors": []
        }

        # Process wake requests first (priority)
        results["links_woken"] = await self.process_wake_queue()

        # Check for links to wake based on predictions
        wake_candidates = self.get_wake_candidates()
        for src_dpid, src_port, dst_dpid, dst_port in wake_candidates:
            success = await self.initiate_wake(
                src_dpid, src_port, dst_dpid, dst_port,
                reason="predicted_load_increase"
            )
            if success:
                results["links_woken"] += 1

        # Check for links to sleep
        sleep_candidates = self.get_sleep_candidates()
        for src_dpid, src_port, dst_dpid, dst_port in sleep_candidates:
            success = await self.initiate_sleep(
                src_dpid, src_port, dst_dpid, dst_port
            )
            if success:
                results["links_slept"] += 1

        return results

    def get_active_transitions(self) -> List[Dict]:
        """Get currently active transitions."""
        return [
            {
                "src_dpid": t.src_dpid,
                "src_port": t.src_port,
                "dst_dpid": t.dst_dpid,
                "dst_port": t.dst_port,
                "state": t.state.value,
                "started_at": t.started_at,
                "flows_rerouted": len(t.flows_rerouted)
            }
            for t in self._transitions.values()
        ]

    def get_stats(self) -> Dict:
        """Get sleep manager statistics."""
        return {
            **self._stats,
            "pending_wake_requests": len(self._wake_queue),
            "active_transitions": len(self._transitions),
            "sleep_candidates": len(self.get_sleep_candidates()),
            "wake_candidates": len(self.get_wake_candidates())
        }

    def reset(self):
        """Reset sleep manager state."""
        self._transitions.clear()
        self._wake_queue.clear()
        self._stats = {
            "sleep_attempts": 0,
            "sleep_successes": 0,
            "sleep_failures": 0,
            "wake_attempts": 0,
            "wake_successes": 0,
            "wake_failures": 0,
            "flows_rerouted": 0,
            "rollbacks": 0
        }
        logger.info("sleep_manager_reset")
