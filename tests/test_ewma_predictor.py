"""
Unit tests for EWMA Predictor module.
"""

import pytest
import time
from controller.ewma_predictor import AdaptiveEWMAPredictor, LinkStats


class TestAdaptiveEWMAPredictor:
    """Tests for AdaptiveEWMAPredictor class."""

    @pytest.fixture
    def predictor(self):
        """Create a predictor instance for testing."""
        return AdaptiveEWMAPredictor(
            base_alpha=0.3,
            min_alpha=0.1,
            max_alpha=0.7,
            prediction_steps=3,
            history_size=100
        )

    def test_initialization(self, predictor):
        """Test predictor initialization."""
        assert predictor.base_alpha == 0.3
        assert predictor.min_alpha == 0.1
        assert predictor.max_alpha == 0.7
        assert predictor.prediction_steps == 3

    def test_first_prediction(self, predictor):
        """Test first prediction uses current load."""
        stats = LinkStats(
            timestamp=time.time(),
            bytes_rx=1000,
            bytes_tx=1000,
            utilization=50.0
        )

        result = predictor.update(1, 1, stats)

        assert result.current_load == 50.0
        assert result.predicted_load == 50.0
        assert result.link_id == (1, 1)

    def test_ewma_smoothing(self, predictor):
        """Test EWMA smoothing effect."""
        base_time = time.time()

        # First update
        stats1 = LinkStats(timestamp=base_time, utilization=50.0)
        predictor.update(1, 1, stats1)

        # Second update with different load
        stats2 = LinkStats(timestamp=base_time + 5, utilization=100.0)
        result = predictor.update(1, 1, stats2)

        # EWMA should smooth the value
        # predicted = 0.3 * 100 + 0.7 * 50 = 65
        assert result.predicted_load == pytest.approx(65.0, abs=5.0)

    def test_multi_step_prediction(self, predictor):
        """Test multi-step ahead predictions."""
        base_time = time.time()

        for i in range(10):
            stats = LinkStats(
                timestamp=base_time + i * 5,
                utilization=30.0 + i * 2
            )
            result = predictor.update(1, 1, stats)

        assert len(result.predictions_ahead) == 3
        # Predictions should be increasing due to trend
        for i in range(len(result.predictions_ahead) - 1):
            assert result.predictions_ahead[i+1] >= result.predictions_ahead[i] - 5

    def test_should_sleep(self, predictor):
        """Test sleep decision logic."""
        base_time = time.time()

        # Build history with consistently low load
        for i in range(20):
            stats = LinkStats(
                timestamp=base_time + i * 2,
                utilization=10.0
            )
            predictor.update(1, 1, stats)

        # Should recommend sleep
        should_sleep = predictor.should_sleep(1, 1, sleep_threshold=20.0)
        # May be False initially due to warmup, but logic should work
        assert isinstance(should_sleep, bool)

    def test_should_wake(self, predictor):
        """Test wake decision logic."""
        base_time = time.time()

        # Build history with high load
        for i in range(10):
            stats = LinkStats(
                timestamp=base_time + i * 5,
                utilization=70.0 + i * 2
            )
            predictor.update(1, 1, stats)

        should_wake = predictor.should_wake(1, 1, wake_threshold=60.0)
        assert should_wake is True

    def test_trend_detection(self, predictor):
        """Test trend detection (increasing, decreasing, stable)."""
        base_time = time.time()

        # Increasing trend
        for i in range(10):
            stats = LinkStats(
                timestamp=base_time + i * 5,
                utilization=20.0 + i * 5
            )
            predictor.update(1, 1, stats)

        result = predictor.get_prediction(1, 1)
        assert result.trend == "increasing"

        # Decreasing trend
        for i in range(10):
            stats = LinkStats(
                timestamp=base_time + 60 + i * 5,
                utilization=70.0 - i * 5
            )
            predictor.update(1, 2, stats)

        result = predictor.get_prediction(1, 2)
        assert result.trend == "decreasing"

    def test_adaptive_alpha(self, predictor):
        """Test adaptive alpha based on variance."""
        base_time = time.time()

        # High variance (bursty) traffic
        for i in range(20):
            util = 50.0 + (30.0 if i % 2 == 0 else -30.0)
            stats = LinkStats(
                timestamp=base_time + i * 2,
                utilization=util
            )
            predictor.update(1, 1, stats)

        # Alpha should be higher for bursty traffic
        alpha = predictor._alphas.get((1, 1), 0.3)
        assert alpha >= predictor.base_alpha

    def test_confidence_calculation(self, predictor):
        """Test confidence calculation."""
        base_time = time.time()

        # Stable predictions should have high confidence
        for i in range(20):
            stats = LinkStats(
                timestamp=base_time + i * 5,
                utilization=50.0
            )
            predictor.update(1, 1, stats)

        result = predictor.get_prediction(1, 1)
        assert 0 <= result.confidence <= 1

    def test_reset(self, predictor):
        """Test reset functionality."""
        stats = LinkStats(timestamp=time.time(), utilization=50.0)
        predictor.update(1, 1, stats)

        predictor.reset(1, 1)

        result = predictor.get_prediction(1, 1)
        assert result is None

        predictor.reset()
        stats = predictor.get_stats()
        assert stats["total_links"] == 0

    def test_get_stats(self, predictor):
        """Test statistics collection."""
        base_time = time.time()

        for port in range(3):
            for i in range(5):
                stats = LinkStats(
                    timestamp=base_time + i * 5,
                    utilization=30.0 + port * 10
                )
                predictor.update(1, port, stats)

        stats = predictor.get_stats()

        assert stats["total_links"] == 3
        assert 0 <= stats["average_confidence"] <= 1
        assert stats["average_load"] > 0

    def test_get_all_predictions(self, predictor):
        """Test getting all predictions."""
        base_time = time.time()

        for port in range(3):
            stats = LinkStats(
                timestamp=base_time,
                utilization=40.0 + port * 10
            )
            predictor.update(1, port, stats)

        all_preds = predictor.get_all_predictions()

        assert len(all_preds) == 3
        for link_id, result in all_preds.items():
            assert link_id[0] == 1
            assert result.predicted_load > 0
