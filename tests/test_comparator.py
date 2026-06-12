"""
test_comparator.py — Unit tests for RunComparator.
"""

from pathlib import Path
import sys
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tracking.run_record import RunRecord
from src.tracking.tracker import ExperimentTracker
from src.tracking.comparator import RunComparator


class MockTracker:
    """Mock tracker that bypasses SQLite database operations for fast comparison testing."""

    def __init__(self, runs: list[RunRecord]) -> None:
        self.runs = runs

    def list_runs(self, limit: int = 1000) -> list[RunRecord]:
        return self.runs[:limit]

    def get_run(self, run_id: str) -> RunRecord:
        for r in self.runs:
            if r.run_id == run_id:
                return r
        raise ValueError(f"Run ID '{run_id}' not found.")


@pytest.fixture()
def mock_tracker() -> MockTracker:
    """Fixture providing a list of mock runs with known metrics."""
    r1 = RunRecord(
        run_id="run_1",
        config={},
        pipeline_summary=[],
        metrics={"f1_weighted": 0.80, "rmse": 2.5, "accuracy": 0.82},
        model_type="logistic_regression",
        model_path="path/1.joblib",
        dataset_path="data.csv",
        train_rows=100,
        test_rows=20,
        duration_seconds=1.0
    )
    r2 = RunRecord(
        run_id="run_2",
        config={},
        pipeline_summary=[],
        metrics={"f1_weighted": 0.90, "rmse": 1.5, "accuracy": 0.91},
        model_type="random_forest",
        model_path="path/2.joblib",
        dataset_path="data.csv",
        train_rows=100,
        test_rows=20,
        duration_seconds=2.0
    )
    r3 = RunRecord(
        run_id="run_3",
        config={},
        pipeline_summary=[],
        metrics={"f1_weighted": 0.85, "rmse": 3.0, "accuracy": 0.87},
        model_type="decision_tree",
        model_path="path/3.joblib",
        dataset_path="data.csv",
        train_rows=100,
        test_rows=20,
        duration_seconds=0.5
    )
    return MockTracker([r1, r2, r3])


class TestRunComparator:
    """Verify ranking, sorting, side-by-side comparison matrix, and formatted outputs."""

    def test_get_best_descending_for_scores(self, mock_tracker):
        """get_best() should sort scores (f1, accuracy) descending (higher is better)."""
        comparator = RunComparator(mock_tracker)
        
        # Rank by F1 (should be run_2 (0.90), run_3 (0.85), run_1 (0.80))
        best_f1 = comparator.get_best("f1", top_n=3)
        assert len(best_f1) == 3
        assert best_f1[0].run_id == "run_2"
        assert best_f1[1].run_id == "run_3"
        assert best_f1[2].run_id == "run_1"

        # Rank by Accuracy
        best_acc = comparator.get_best("accuracy", top_n=2)
        assert len(best_acc) == 2
        assert best_acc[0].run_id == "run_2"
        assert best_acc[1].run_id == "run_3"

    def test_get_best_ascending_for_errors(self, mock_tracker):
        """get_best() should sort error metrics (rmse, mae) ascending (lower is better)."""
        comparator = RunComparator(mock_tracker)

        # Rank by rmse (should be run_2 (1.5), run_1 (2.5), run_3 (3.0))
        best_rmse = comparator.get_best("rmse", top_n=3)
        assert len(best_rmse) == 3
        assert best_rmse[0].run_id == "run_2"
        assert best_rmse[1].run_id == "run_1"
        assert best_rmse[2].run_id == "run_3"

    def test_compare_returns_dataframe(self, mock_tracker):
        """compare() must return a formatted pandas DataFrame with aligned comparison metrics."""
        comparator = RunComparator(mock_tracker)
        
        df = comparator.compare(["run_1", "run_2"])
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 2  # 2 rows for 2 runs
        
        # Checking that expected key columns and metrics are present
        assert "run_id" in df.columns
        assert "model_type" in df.columns
        assert "f1_weighted" in df.columns
        assert "rmse" in df.columns

        # Verify data values
        row1 = df[df["run_id"] == "run_1"].iloc[0]
        assert row1["model_type"] == "logistic_regression"
        assert row1["f1_weighted"] == 0.80
        assert row1["rmse"] == 2.5

    def test_summary_table(self, mock_tracker):
        """summary_table() must return a formatted ASCII table string containing rank and short run IDs."""
        comparator = RunComparator(mock_tracker)
        table_str = comparator.summary_table(top_n=3)
        assert isinstance(table_str, str)
        # Should contain headers
        assert "Rank" in table_str
        assert "Run ID" in table_str
        assert "Model" in table_str
        # Should contain run_2 as top rank
        assert "run_2"[:8] in table_str
