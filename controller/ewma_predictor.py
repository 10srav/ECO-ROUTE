"""
EWMA Predictor Module for EcoRoute SDN Controller

Implements Exponentially Weighted Moving Average (EWMA) traffic prediction
with adaptive smoothing factor for accurate load forecasting.

Formula: predicted_load_t = α * current_load_t + (1-α) * predicted_load_(t-1)
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class LinkStats:
    """Statistics for a single link/port."""
    timestamp: float
    bytes_rx: int = 0
    bytes_tx: int = 0
    packets_rx: int = 0
    packets_tx: int = 0
    errors: int = 0
    utilization: float = 0.0  # Percentage (0-100)


@dataclass
class PredictionResult:
    """Result of EWMA prediction for a link."""
    link_id: Tuple[int, int]  # (dpid, port_no)
    current_load: float
    predicted_load: float
    predictions_ahead: List[float]  # Multi-step predictions
    confidence: float  # Prediction confidence (0-1)
    trend: str  # 'increasing', 'decreasing', 'stable'
    timestamp: float = field(default_factory=time.time)


class AdaptiveEWMAPredictor:
    """
    Adaptive EWMA Traffic Predictor

    Features:
    - Adaptive alpha based on traffic variance (burst detection)
    - Multi-step ahead prediction for wake-up latency coverage
    - Per-link prediction with history tracking
    - Confidence estimation based on prediction accuracy
    """

    def __init__(
        self,
        base_alpha: float = 0.3,
        min_alpha: float = 0.1,
        max_alpha: float = 0.7,
        burst_threshold: float = 0.3,
        prediction_steps: int = 3,
        history_size: int = 100,
        prediction_window: float = 30.0
    ):
        """
        Initialize EWMA Predictor.

        Args:
            base_alpha: Base smoothing factor (0 < alpha < 1)
            min_alpha: Minimum alpha for stable traffic
            max_alpha: Maximum alpha for bursty traffic
            burst_threshold: Variance threshold for burst detection
            prediction_steps: Number of steps to predict ahead
            history_size: Maximum history entries per link
            prediction_window: Time window for predictions (seconds)
        """
        self.base_alpha = base_alpha
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.burst_threshold = burst_threshold
        self.prediction_steps = prediction_steps
        self.history_size = history_size
        self.prediction_window = prediction_window

        # Per-link state tracking
        # Key: (dpid, port_no), Value: deque of LinkStats
        self._history: Dict[Tuple[int, int], deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

        # Current predictions per link
        self._predictions: Dict[Tuple[int, int], float] = defaultdict(float)

        # Adaptive alpha per link
        self._alphas: Dict[Tuple[int, int], float] = defaultdict(
            lambda: base_alpha
        )

        # Prediction errors for confidence estimation
        self._errors: Dict[Tuple[int, int], deque] = defaultdict(
            lambda: deque(maxlen=20)
        )

        logger.info(
            "ewma_predictor_initialized",
            base_alpha=base_alpha,
            prediction_steps=prediction_steps,
            history_size=history_size
        )

    def update(
        self,
        dpid: int,
        port_no: int,
        stats: LinkStats,
        bandwidth_capacity: float = 1000.0  # Mbps
    ) -> PredictionResult:
        """
        Update prediction with new stats and return prediction result.

        Args:
            dpid: Datapath ID of the switch
            port_no: Port number
            stats: Current link statistics
            bandwidth_capacity: Link capacity in Mbps

        Returns:
            PredictionResult with current and predicted loads
        """
        link_id = (dpid, port_no)
        history = self._history[link_id]

        # Calculate current utilization if not provided
        if stats.utilization == 0.0 and len(history) > 0:
            prev_stats = history[-1]
            time_delta = stats.timestamp - prev_stats.timestamp
            if time_delta > 0:
                bytes_delta = (
                    (stats.bytes_rx + stats.bytes_tx) -
                    (prev_stats.bytes_rx + prev_stats.bytes_tx)
                )
                # Convert to Mbps and calculate utilization percentage
                throughput_mbps = (bytes_delta * 8) / (time_delta * 1_000_000)
                stats.utilization = min(
                    100.0,
                    (throughput_mbps / bandwidth_capacity) * 100
                )

        # Add to history
        history.append(stats)

        current_load = stats.utilization

        # Calculate prediction error from previous prediction
        prev_prediction = self._predictions[link_id]
        if prev_prediction > 0:
            error = abs(current_load - prev_prediction) / max(prev_prediction, 1.0)
            self._errors[link_id].append(error)

        # Adapt alpha based on traffic variance (burst detection)
        alpha = self._adapt_alpha(link_id, current_load)
        self._alphas[link_id] = alpha

        # EWMA prediction: predicted = α * current + (1-α) * previous_prediction
        if prev_prediction == 0:
            # First prediction - use current load
            predicted_load = current_load
        else:
            predicted_load = alpha * current_load + (1 - alpha) * prev_prediction

        self._predictions[link_id] = predicted_load

        # Multi-step ahead predictions
        predictions_ahead = self._predict_ahead(link_id, predicted_load, alpha)

        # Calculate confidence based on historical accuracy
        confidence = self._calculate_confidence(link_id)

        # Determine trend
        trend = self._determine_trend(link_id)

        result = PredictionResult(
            link_id=link_id,
            current_load=current_load,
            predicted_load=predicted_load,
            predictions_ahead=predictions_ahead,
            confidence=confidence,
            trend=trend
        )

        logger.debug(
            "ewma_prediction_updated",
            dpid=dpid,
            port_no=port_no,
            current_load=round(current_load, 2),
            predicted_load=round(predicted_load, 2),
            alpha=round(alpha, 3),
            confidence=round(confidence, 2),
            trend=trend
        )

        return result

    def _adapt_alpha(self, link_id: Tuple[int, int], current_load: float) -> float:
        """
        Adapt alpha based on traffic variance for burst detection.

        Higher variance (bursty traffic) -> higher alpha (more reactive)
        Lower variance (stable traffic) -> lower alpha (more smooth)
        """
        history = self._history[link_id]

        if len(history) < 5:
            return self.base_alpha

        # Calculate recent variance
        recent_loads = [s.utilization for s in list(history)[-10:]]
        variance = np.var(recent_loads) if len(recent_loads) > 1 else 0

        # Normalize variance
        normalized_variance = min(1.0, variance / 100.0)

        # Adapt alpha: higher variance -> higher alpha
        if normalized_variance > self.burst_threshold:
            # Bursty traffic - increase alpha for faster response
            alpha = min(
                self.max_alpha,
                self.base_alpha + (self.max_alpha - self.base_alpha) *
                (normalized_variance / self.burst_threshold)
            )
        else:
            # Stable traffic - use lower alpha for smoothing
            alpha = max(
                self.min_alpha,
                self.base_alpha - (self.base_alpha - self.min_alpha) *
                (1 - normalized_variance / self.burst_threshold)
            )

        return alpha

    def _predict_ahead(
        self,
        link_id: Tuple[int, int],
        current_prediction: float,
        alpha: float
    ) -> List[float]:
        """
        Generate multi-step ahead predictions.

        Uses trend extrapolation combined with EWMA for future predictions.
        """
        predictions = []
        history = self._history[link_id]

        if len(history) < 3:
            # Not enough history - assume stable
            return [current_prediction] * self.prediction_steps

        # Calculate trend from recent history
        recent_loads = [s.utilization for s in list(history)[-5:]]
        if len(recent_loads) >= 2:
            # Linear trend estimation
            trend_slope = (recent_loads[-1] - recent_loads[0]) / len(recent_loads)
        else:
            trend_slope = 0.0

        # Generate predictions
        pred = current_prediction
        for step in range(1, self.prediction_steps + 1):
            # Combine EWMA smoothing with trend
            trend_contribution = trend_slope * step * 0.5  # Damped trend
            pred = pred + trend_contribution
            # Clamp to valid range
            pred = max(0.0, min(100.0, pred))
            predictions.append(pred)

        return predictions

    def _calculate_confidence(self, link_id: Tuple[int, int]) -> float:
        """
        Calculate prediction confidence based on historical accuracy.

        Returns value between 0 (low confidence) and 1 (high confidence).
        """
        errors = self._errors[link_id]

        if len(errors) < 5:
            return 0.5  # Default confidence with limited history

        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(list(errors))

        # Convert MAPE to confidence (lower error -> higher confidence)
        # MAPE of 0 -> confidence 1.0
        # MAPE of 0.5+ -> confidence 0.5
        confidence = max(0.3, min(1.0, 1.0 - mape))

        return confidence

    def _determine_trend(self, link_id: Tuple[int, int]) -> str:
        """Determine traffic trend from recent history."""
        history = self._history[link_id]

        if len(history) < 5:
            return "stable"

        recent_loads = [s.utilization for s in list(history)[-10:]]

        # Simple trend detection using linear regression slope
        x = np.arange(len(recent_loads))
        slope = np.polyfit(x, recent_loads, 1)[0]

        # Threshold for trend classification
        if slope > 2.0:
            return "increasing"
        elif slope < -2.0:
            return "decreasing"
        else:
            return "stable"

    def get_prediction(self, dpid: int, port_no: int) -> Optional[PredictionResult]:
        """Get the latest prediction for a link without updating."""
        link_id = (dpid, port_no)

        if link_id not in self._predictions:
            return None

        history = self._history[link_id]
        if not history:
            return None

        current_load = history[-1].utilization
        predicted_load = self._predictions[link_id]
        alpha = self._alphas[link_id]

        return PredictionResult(
            link_id=link_id,
            current_load=current_load,
            predicted_load=predicted_load,
            predictions_ahead=self._predict_ahead(link_id, predicted_load, alpha),
            confidence=self._calculate_confidence(link_id),
            trend=self._determine_trend(link_id)
        )

    def get_all_predictions(self) -> Dict[Tuple[int, int], PredictionResult]:
        """Get predictions for all tracked links."""
        results = {}
        for link_id in self._predictions:
            dpid, port_no = link_id
            result = self.get_prediction(dpid, port_no)
            if result:
                results[link_id] = result
        return results

    def should_sleep(
        self,
        dpid: int,
        port_no: int,
        sleep_threshold: float = 20.0,
        min_duration: float = 30.0
    ) -> bool:
        """
        Determine if a link should be put to sleep based on predictions.

        Args:
            dpid: Datapath ID
            port_no: Port number
            sleep_threshold: Load threshold below which to sleep (%)
            min_duration: Minimum time load should stay below threshold (s)

        Returns:
            True if link should sleep, False otherwise
        """
        link_id = (dpid, port_no)

        prediction = self.get_prediction(dpid, port_no)
        if not prediction:
            return False

        # Check if current and all future predictions are below threshold
        if prediction.predicted_load >= sleep_threshold:
            return False

        for future_load in prediction.predictions_ahead:
            if future_load >= sleep_threshold:
                return False

        # Check historical stability (load below threshold for min_duration)
        history = self._history[link_id]
        if len(history) < 3:
            return False

        # Check recent history
        current_time = time.time()
        stable_duration = 0.0

        for stats in reversed(list(history)):
            if stats.utilization < sleep_threshold:
                stable_duration = current_time - stats.timestamp
            else:
                break

        # Require minimum stable duration before sleeping
        if stable_duration < min_duration * 0.5:  # Half the duration as warmup
            return False

        # High confidence in prediction
        if prediction.confidence < 0.6:
            return False

        logger.info(
            "link_eligible_for_sleep",
            dpid=dpid,
            port_no=port_no,
            predicted_load=round(prediction.predicted_load, 2),
            confidence=round(prediction.confidence, 2)
        )

        return True

    def should_wake(
        self,
        dpid: int,
        port_no: int,
        wake_threshold: float = 60.0
    ) -> bool:
        """
        Determine if a sleeping link should be woken up.

        Args:
            dpid: Datapath ID
            port_no: Port number
            wake_threshold: Load threshold above which to wake (%)

        Returns:
            True if link should wake, False otherwise
        """
        prediction = self.get_prediction(dpid, port_no)
        if not prediction:
            return False

        # Wake if any prediction exceeds threshold
        if prediction.predicted_load > wake_threshold:
            return True

        for future_load in prediction.predictions_ahead:
            if future_load > wake_threshold:
                return True

        # Wake if trend is strongly increasing
        if prediction.trend == "increasing" and prediction.confidence > 0.7:
            return True

        return False

    def reset(self, dpid: Optional[int] = None, port_no: Optional[int] = None):
        """Reset prediction state for a specific link or all links."""
        if dpid is not None and port_no is not None:
            link_id = (dpid, port_no)
            self._history.pop(link_id, None)
            self._predictions.pop(link_id, None)
            self._alphas.pop(link_id, None)
            self._errors.pop(link_id, None)
            logger.info("ewma_link_reset", dpid=dpid, port_no=port_no)
        else:
            self._history.clear()
            self._predictions.clear()
            self._alphas.clear()
            self._errors.clear()
            logger.info("ewma_predictor_reset")

    def get_stats(self) -> Dict:
        """Get predictor statistics for monitoring."""
        total_links = len(self._predictions)

        if total_links == 0:
            return {
                "total_links": 0,
                "average_confidence": 0.0,
                "average_load": 0.0
            }

        confidences = [
            self._calculate_confidence(link_id)
            for link_id in self._predictions
        ]

        loads = [
            self._predictions[link_id]
            for link_id in self._predictions
        ]

        return {
            "total_links": total_links,
            "average_confidence": round(np.mean(confidences), 3),
            "average_load": round(np.mean(loads), 2),
            "max_load": round(max(loads), 2) if loads else 0.0,
            "min_load": round(min(loads), 2) if loads else 0.0,
            "links_below_20": sum(1 for l in loads if l < 20),
            "links_above_80": sum(1 for l in loads if l > 80)
        }
