#!/usr/bin/env python3
"""
EWMA Model Training Pipeline for EcoRoute

Trains and optimizes the Adaptive EWMA predictor using:
1. Grid search for optimal alpha parameters
2. Cross-validation on traffic traces
3. Performance metrics (MAPE, RMSE, MAE)
4. Saves trained model parameters

Target: Minimize prediction error while maintaining responsiveness to traffic changes.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from controller.ewma_predictor import AdaptiveEWMAPredictor, LinkStats


@dataclass
class TrainingMetrics:
    """Metrics from training run."""
    alpha: float
    min_alpha: float
    max_alpha: float
    mape: float  # Mean Absolute Percentage Error
    rmse: float  # Root Mean Square Error
    mae: float   # Mean Absolute Error
    prediction_accuracy: float
    burst_response_time: float  # Steps to respond to burst
    samples_evaluated: int


@dataclass
class TrainedModel:
    """Trained model parameters."""
    optimal_alpha: float
    optimal_min_alpha: float
    optimal_max_alpha: float
    burst_threshold: float
    prediction_steps: int
    metrics: TrainingMetrics
    trained_at: str
    training_duration_sec: float
    dataset_info: Dict


class EWMATrainer:
    """
    EWMA Model Trainer

    Optimizes alpha parameters using grid search with cross-validation.
    Evaluates on multiple metrics to find best balance between
    smoothing (low alpha) and responsiveness (high alpha).
    """

    def __init__(
        self,
        alpha_range: Tuple[float, float] = (0.1, 0.9),
        alpha_step: float = 0.05,
        min_alpha_range: Tuple[float, float] = (0.05, 0.3),
        max_alpha_range: Tuple[float, float] = (0.5, 0.9),
        cv_folds: int = 5
    ):
        """
        Initialize trainer.

        Args:
            alpha_range: Range for base alpha search
            alpha_step: Step size for grid search
            min_alpha_range: Range for minimum alpha (stable traffic)
            max_alpha_range: Range for maximum alpha (bursty traffic)
            cv_folds: Number of cross-validation folds
        """
        self.alpha_range = alpha_range
        self.alpha_step = alpha_step
        self.min_alpha_range = min_alpha_range
        self.max_alpha_range = max_alpha_range
        self.cv_folds = cv_folds

        self.results: List[TrainingMetrics] = []

    def load_training_data(self, filepath: str) -> List[LinkUtilizationRecord]:
        """Load link utilization data from CSV."""
        records = []

        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(LinkUtilizationRecord(
                    timestamp=float(row['timestamp']),
                    link_id=row['link_id'],
                    utilization=float(row['utilization_percent'])
                ))

        return records

    def prepare_time_series(
        self,
        records: List[LinkUtilizationRecord]
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Organize records into per-link time series.

        Returns:
            Dict mapping link_id to list of (timestamp, utilization) tuples
        """
        time_series = {}

        for record in records:
            if record.link_id not in time_series:
                time_series[record.link_id] = []
            time_series[record.link_id].append(
                (record.timestamp, record.utilization)
            )

        # Sort by timestamp
        for link_id in time_series:
            time_series[link_id].sort(key=lambda x: x[0])

        return time_series

    def evaluate_predictor(
        self,
        predictor: AdaptiveEWMAPredictor,
        time_series: List[Tuple[float, float]],
        warmup_samples: int = 10
    ) -> Tuple[float, float, float, float]:
        """
        Evaluate predictor on a time series.

        Returns:
            (MAPE, RMSE, MAE, accuracy)
        """
        errors = []
        squared_errors = []
        abs_errors = []
        correct_predictions = 0
        total_predictions = 0

        # Reset predictor
        predictor.reset()

        for i, (timestamp, actual_util) in enumerate(time_series):
            # Create stats
            stats = LinkStats(
                timestamp=timestamp,
                utilization=actual_util
            )

            # Get prediction before updating (if we have history)
            if i >= warmup_samples:
                prediction = predictor.get_prediction(1, 1)

                if prediction:
                    predicted = prediction.predicted_load
                    error = abs(actual_util - predicted)
                    pct_error = error / max(actual_util, 1.0) * 100

                    errors.append(pct_error)
                    squared_errors.append(error ** 2)
                    abs_errors.append(error)

                    # Count as correct if within 10%
                    if pct_error < 10:
                        correct_predictions += 1
                    total_predictions += 1

            # Update predictor
            predictor.update(1, 1, stats)

        if not errors:
            return 100.0, 100.0, 100.0, 0.0

        mape = np.mean(errors)
        rmse = np.sqrt(np.mean(squared_errors))
        mae = np.mean(abs_errors)
        accuracy = correct_predictions / total_predictions * 100 if total_predictions > 0 else 0

        return mape, rmse, mae, accuracy

    def evaluate_burst_response(
        self,
        predictor: AdaptiveEWMAPredictor,
        time_series: List[Tuple[float, float]]
    ) -> float:
        """
        Evaluate how quickly predictor responds to traffic bursts.

        Returns:
            Average steps to respond to burst
        """
        predictor.reset()

        burst_responses = []
        in_burst = False
        burst_start_idx = 0
        prev_util = 0

        for i, (timestamp, actual_util) in enumerate(time_series):
            stats = LinkStats(timestamp=timestamp, utilization=actual_util)
            predictor.update(1, 1, stats)

            # Detect burst start (>50% increase)
            if not in_burst and prev_util > 0:
                if actual_util > prev_util * 1.5:
                    in_burst = True
                    burst_start_idx = i

            # Check if predictor caught up to burst
            if in_burst:
                prediction = predictor.get_prediction(1, 1)
                if prediction and prediction.predicted_load >= actual_util * 0.9:
                    burst_responses.append(i - burst_start_idx)
                    in_burst = False

            prev_util = actual_util

        return np.mean(burst_responses) if burst_responses else 10.0

    def grid_search(
        self,
        time_series_dict: Dict[str, List[Tuple[float, float]]],
        progress_callback=None
    ) -> TrainingMetrics:
        """
        Perform grid search over alpha parameters.

        Args:
            time_series_dict: Per-link time series data
            progress_callback: Optional callback for progress updates

        Returns:
            Best TrainingMetrics
        """
        best_metrics = None
        best_score = float('inf')

        # Generate parameter combinations
        alphas = np.arange(
            self.alpha_range[0],
            self.alpha_range[1] + self.alpha_step,
            self.alpha_step
        )

        total_combinations = len(alphas) * 3 * 3  # Simplified: 3 min, 3 max options
        current = 0

        for alpha in alphas:
            for min_alpha in [0.1, 0.15, 0.2]:
                for max_alpha in [0.6, 0.7, 0.8]:
                    if min_alpha >= alpha or max_alpha <= alpha:
                        continue

                    current += 1
                    if progress_callback:
                        progress_callback(current, total_combinations)

                    # Create predictor with these parameters
                    predictor = AdaptiveEWMAPredictor(
                        base_alpha=alpha,
                        min_alpha=min_alpha,
                        max_alpha=max_alpha,
                        prediction_steps=3
                    )

                    # Evaluate on all links
                    all_mape = []
                    all_rmse = []
                    all_mae = []
                    all_accuracy = []
                    all_burst_response = []

                    for link_id, series in time_series_dict.items():
                        if len(series) < 100:
                            continue

                        mape, rmse, mae, acc = self.evaluate_predictor(
                            predictor, series
                        )
                        burst_resp = self.evaluate_burst_response(
                            predictor, series
                        )

                        all_mape.append(mape)
                        all_rmse.append(rmse)
                        all_mae.append(mae)
                        all_accuracy.append(acc)
                        all_burst_response.append(burst_resp)

                    if not all_mape:
                        continue

                    metrics = TrainingMetrics(
                        alpha=alpha,
                        min_alpha=min_alpha,
                        max_alpha=max_alpha,
                        mape=np.mean(all_mape),
                        rmse=np.mean(all_rmse),
                        mae=np.mean(all_mae),
                        prediction_accuracy=np.mean(all_accuracy),
                        burst_response_time=np.mean(all_burst_response),
                        samples_evaluated=sum(len(s) for s in time_series_dict.values())
                    )

                    self.results.append(metrics)

                    # Combined score (lower is better)
                    # Weight MAPE highly, but also consider burst response
                    score = metrics.mape * 0.7 + metrics.burst_response_time * 0.3

                    if score < best_score:
                        best_score = score
                        best_metrics = metrics

        return best_metrics

    def cross_validate(
        self,
        time_series_dict: Dict[str, List[Tuple[float, float]]],
        alpha: float,
        min_alpha: float,
        max_alpha: float
    ) -> Tuple[float, float]:
        """
        Cross-validate parameters.

        Returns:
            (mean_mape, std_mape)
        """
        fold_scores = []

        for link_id, series in time_series_dict.items():
            if len(series) < 100:
                continue

            # Split into folds
            fold_size = len(series) // self.cv_folds

            for fold in range(self.cv_folds):
                # Use this fold as test
                test_start = fold * fold_size
                test_end = test_start + fold_size
                test_data = series[test_start:test_end]

                # Train on rest
                train_data = series[:test_start] + series[test_end:]

                predictor = AdaptiveEWMAPredictor(
                    base_alpha=alpha,
                    min_alpha=min_alpha,
                    max_alpha=max_alpha
                )

                # Warm up on training data
                for ts, util in train_data[-50:]:
                    stats = LinkStats(timestamp=ts, utilization=util)
                    predictor.update(1, 1, stats)

                # Evaluate on test
                mape, _, _, _ = self.evaluate_predictor(predictor, test_data, warmup_samples=0)
                fold_scores.append(mape)

        return np.mean(fold_scores), np.std(fold_scores)

    def train(
        self,
        data_path: str,
        output_dir: str = "training/models"
    ) -> TrainedModel:
        """
        Full training pipeline.

        Args:
            data_path: Path to link_utilization.csv
            output_dir: Directory to save trained model

        Returns:
            TrainedModel with optimized parameters
        """
        start_time = time.time()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("EWMA Model Training Pipeline")
        print("=" * 60)

        # Load data
        print(f"\n1. Loading training data from {data_path}...")
        records = self.load_training_data(data_path)
        print(f"   Loaded {len(records)} samples")

        # Prepare time series
        print("\n2. Preparing time series...")
        time_series = self.prepare_time_series(records)
        print(f"   Found {len(time_series)} links")
        print(f"   Average series length: {np.mean([len(s) for s in time_series.values()]):.0f}")

        # Grid search
        print("\n3. Running grid search optimization...")

        def progress(current, total):
            pct = current / total * 100
            print(f"\r   Progress: {pct:.1f}% ({current}/{total})", end="", flush=True)

        best_metrics = self.grid_search(time_series, progress_callback=progress)
        print("\n")

        if best_metrics is None:
            raise ValueError("Training failed - no valid results")

        # Cross-validate best parameters
        print("4. Cross-validating best parameters...")
        cv_mean, cv_std = self.cross_validate(
            time_series,
            best_metrics.alpha,
            best_metrics.min_alpha,
            best_metrics.max_alpha
        )
        print(f"   CV MAPE: {cv_mean:.2f}% (+/- {cv_std:.2f}%)")

        # Create trained model
        training_duration = time.time() - start_time

        trained_model = TrainedModel(
            optimal_alpha=best_metrics.alpha,
            optimal_min_alpha=best_metrics.min_alpha,
            optimal_max_alpha=best_metrics.max_alpha,
            burst_threshold=0.3,
            prediction_steps=3,
            metrics=best_metrics,
            trained_at=datetime.now().isoformat(),
            training_duration_sec=training_duration,
            dataset_info={
                "path": data_path,
                "total_samples": len(records),
                "num_links": len(time_series),
                "cv_folds": self.cv_folds
            }
        )

        # Save model
        model_path = output_path / "ewma_model.json"
        self._save_model(trained_model, model_path)

        # Print results
        print("\n" + "=" * 60)
        print("TRAINING RESULTS")
        print("=" * 60)
        print(f"\nOptimal Parameters:")
        print(f"  Base Alpha:  {trained_model.optimal_alpha:.3f}")
        print(f"  Min Alpha:   {trained_model.optimal_min_alpha:.3f}")
        print(f"  Max Alpha:   {trained_model.optimal_max_alpha:.3f}")

        print(f"\nPerformance Metrics:")
        print(f"  MAPE:        {best_metrics.mape:.2f}%")
        print(f"  RMSE:        {best_metrics.rmse:.2f}")
        print(f"  MAE:         {best_metrics.mae:.2f}")
        print(f"  Accuracy:    {best_metrics.prediction_accuracy:.1f}%")
        print(f"  Burst Resp:  {best_metrics.burst_response_time:.1f} steps")

        print(f"\nTraining Time: {training_duration:.1f}s")
        print(f"Model saved to: {model_path}")

        return trained_model

    def _save_model(self, model: TrainedModel, filepath: Path):
        """Save trained model to JSON."""
        data = {
            "optimal_alpha": model.optimal_alpha,
            "optimal_min_alpha": model.optimal_min_alpha,
            "optimal_max_alpha": model.optimal_max_alpha,
            "burst_threshold": model.burst_threshold,
            "prediction_steps": model.prediction_steps,
            "trained_at": model.trained_at,
            "training_duration_sec": model.training_duration_sec,
            "dataset_info": model.dataset_info,
            "metrics": {
                "mape": model.metrics.mape,
                "rmse": model.metrics.rmse,
                "mae": model.metrics.mae,
                "prediction_accuracy": model.metrics.prediction_accuracy,
                "burst_response_time": model.metrics.burst_response_time,
                "samples_evaluated": model.metrics.samples_evaluated
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


@dataclass
class LinkUtilizationRecord:
    """Single link utilization record."""
    timestamp: float
    link_id: str
    utilization: float


def load_trained_model(filepath: str) -> Dict:
    """Load trained model parameters."""
    with open(filepath, 'r') as f:
        return json.load(f)


def main():
    """Main training entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train EWMA Model")
    parser.add_argument(
        "--data",
        type=str,
        default="data/training/link_utilization.csv",
        help="Path to training data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training/models",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate training data first"
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=48.0,
        help="Hours of data to generate"
    )

    args = parser.parse_args()

    # Generate data if requested or if it doesn't exist
    if args.generate or not os.path.exists(args.data):
        print("Generating training dataset...")
        from data.dataset_loader import generate_training_dataset
        generate_training_dataset(
            output_dir="data/training",
            duration_hours=args.hours,
            seed=42
        )
        print()

    # Train model
    trainer = EWMATrainer(
        alpha_range=(0.1, 0.7),
        alpha_step=0.05,
        cv_folds=5
    )

    trained_model = trainer.train(
        data_path=args.data,
        output_dir=args.output
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
