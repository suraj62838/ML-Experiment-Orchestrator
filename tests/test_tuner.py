"""
test_tuner.py — Unit tests for HyperparameterTuner.
"""

from pathlib import Path
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tracking.tracker import ExperimentTracker
from src.tuning.tuner import HyperparameterTuner


@pytest.fixture()
def memory_tracker() -> ExperimentTracker:
    """Fixture providing an ExperimentTracker tied to an in-memory SQLite database."""
    return ExperimentTracker(db_path=":memory:")


@pytest.fixture()
def tuning_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Fixture providing small fast training and validation classification data."""
    rng = np.random.default_rng(42)
    # Simple linearly separable data
    X_train = pd.DataFrame(rng.standard_normal((50, 2)), columns=["f1", "f2"])
    # y is directly related to f1
    y_train = pd.Series((X_train["f1"] > 0).astype(int))

    X_val = pd.DataFrame(rng.standard_normal((20, 2)), columns=["f1", "f2"])
    y_val = pd.Series((X_val["f1"] > 0).astype(int))

    return X_train, y_train, X_val, y_val


class TestHyperparameterTuner:
    """Verifies optimization loop execution, database logging integration, and study metrics."""

    def test_tuning_runs_trials_and_returns_record(self, memory_tracker, tuning_data):
        """Verifies the tuner executes the requested number of trials and returns the best RunRecord."""
        X_train, y_train, X_val, y_val = tuning_data

        config = {
            "data": {"filepath": "dummy.csv"},
            "model": {
                "type": "logistic_regression",
                "params": {}
            }
        }

        tuner = HyperparameterTuner(
            config=config,
            n_trials=5,
            metric="accuracy",
            direction="maximize",
            tracker=memory_tracker
        )

        best_record = tuner.tune(X_train, y_train, X_val, y_val)

        # Check return value
        assert best_record is not None
        assert best_record.model_type == "logistic_regression"
        assert "accuracy" in best_record.metrics

        # Verify study status
        assert tuner.study is not None
        assert len(tuner.study.trials) == 5

        # Check that study summary has 5 trials
        summary_df = tuner.get_study_summary()
        assert len(summary_df) == 5
        assert "trial" in summary_df.columns
        assert "value" in summary_df.columns
        assert "C" in summary_df.columns
        assert "max_iter" in summary_df.columns

    def test_tuning_improves_or_equals_median(self, memory_tracker, tuning_data):
        """Verify that the best trial metric is greater than or equal to the median of trial metrics."""
        X_train, y_train, X_val, y_val = tuning_data

        config = {
            "data": {"filepath": "dummy.csv"},
            "model": {
                "type": "logistic_regression",
                "params": {}
            }
        }

        tuner = HyperparameterTuner(
            config=config,
            n_trials=10,
            metric="accuracy",
            tracker=memory_tracker
        )

        best_record = tuner.tune(X_train, y_train, X_val, y_val)
        best_val = best_record.metrics["accuracy"]

        # Collect all trial metric values
        all_vals = [t.value for t in tuner.study.trials if t.state.name == "COMPLETE"]
        median_val = np.median(all_vals)

        # In maximization, best should be >= median
        assert best_val >= median_val

    def test_all_trials_logged_to_tracker(self, memory_tracker, tuning_data):
        """Verify that every completed trial gets saved in the ExperimentTracker database."""
        X_train, y_train, X_val, y_val = tuning_data

        config = {
            "data": {"filepath": "dummy.csv"},
            "model": {
                "type": "logistic_regression",
                "params": {}
            }
        }

        # Clear memory tracker by recreating it
        tracker = ExperimentTracker(db_path=":memory:")

        tuner = HyperparameterTuner(
            config=config,
            n_trials=4,
            metric="accuracy",
            tracker=tracker
        )

        _ = tuner.tune(X_train, y_train, X_val, y_val)

        # Retrieve runs from tracker
        logged_runs = tracker.list_runs(limit=100)
        
        # Verify 4 trials are logged
        assert len(logged_runs) == 4
        for run in logged_runs:
            assert run.tags["type"] == "trial"
            assert "trial_number" in run.tags
