"""
test_tracker.py — Unit tests for ExperimentTracker using in-memory SQLite.
"""

from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tracking.run_record import RunRecord
from src.tracking.tracker import ExperimentTracker


@pytest.fixture()
def memory_tracker() -> ExperimentTracker:
    """Fixture providing an ExperimentTracker tied to an in-memory database."""
    # SQLite ":memory:" database
    return ExperimentTracker(db_path=":memory:")


@pytest.fixture()
def sample_record() -> RunRecord:
    """Fixture providing a baseline RunRecord."""
    return RunRecord(
        config={"model": {"type": "logistic_regression", "params": {"C": 1.0}}},
        pipeline_summary=[{"name": "standard_scale", "config": {}}],
        metrics={"f1_weighted": 0.85, "accuracy": 0.86},
        model_type="logistic_regression",
        model_path="outputs/model_dummy.joblib",
        dataset_path="data/dummy.csv",
        train_rows=100,
        test_rows=20,
        duration_seconds=1.5,
        tags={"env": "test"}
    )


class TestTrackerOperations:
    """Tests for core database operations: logging, loading, deleting, and champion promotion."""

    def test_log_and_get_run(self, memory_tracker, sample_record):
        """Verifies that a record is saved to the database and can be retrieved exactly."""
        run_id = memory_tracker.log_run(sample_record)
        assert run_id == sample_record.run_id

        retrieved = memory_tracker.get_run(run_id)
        assert retrieved.run_id == sample_record.run_id
        assert retrieved.model_type == sample_record.model_type
        assert retrieved.train_rows == sample_record.train_rows
        assert retrieved.metrics["f1_weighted"] == 0.85
        assert retrieved.tags["env"] == "test"
        assert retrieved.is_champion is False

    def test_get_run_missing_raises_error(self, memory_tracker):
        """get_run() should raise ValueError if the run ID does not exist."""
        with pytest.raises(ValueError, match="not found"):
            memory_tracker.get_run("non_existent_id")

    def test_list_runs_timestamp_descending(self, memory_tracker, sample_record):
        """list_runs() should return all records sorted by timestamp descending."""
        import time
        from datetime import datetime

        r1 = sample_record
        r1.run_id = "run_1"
        r1.timestamp = datetime(2026, 6, 1, 12, 0, 0).isoformat()

        # Deep copy/recreate r2 to have later timestamp
        r2 = RunRecord(
            run_id="run_2",
            config=r1.config,
            pipeline_summary=r1.pipeline_summary,
            metrics=r1.metrics,
            model_type=r1.model_type,
            model_path=r1.model_path,
            dataset_path=r1.dataset_path,
            train_rows=r1.train_rows,
            test_rows=r1.test_rows,
            duration_seconds=r1.duration_seconds,
            tags=r1.tags
        )
        r2.timestamp = datetime(2026, 6, 2, 12, 0, 0).isoformat()

        memory_tracker.log_run(r1)
        memory_tracker.log_run(r2)

        runs = memory_tracker.list_runs()
        assert len(runs) == 2
        # r2 has later timestamp, should be first
        assert runs[0].run_id == "run_2"
        assert runs[1].run_id == "run_1"

    def test_delete_run(self, memory_tracker, sample_record):
        """delete_run() should successfully remove a run record from the database."""
        memory_tracker.log_run(sample_record)
        # Check it exists
        assert memory_tracker.get_run(sample_record.run_id) is not None

        memory_tracker.delete_run(sample_record.run_id)
        with pytest.raises(ValueError, match="not found"):
            memory_tracker.get_run(sample_record.run_id)

    def test_set_and_get_champion(self, memory_tracker, sample_record):
        """set_champion() must set is_champion=1 for the run and clear it for others."""
        # Log first run
        r1 = sample_record
        r1.run_id = "run_1"
        memory_tracker.log_run(r1)

        # Log second run
        r2 = RunRecord(
            run_id="run_2",
            config=r1.config,
            pipeline_summary=r1.pipeline_summary,
            metrics=r1.metrics,
            model_type=r1.model_type,
            model_path=r1.model_path,
            dataset_path=r1.dataset_path,
            train_rows=r1.train_rows,
            test_rows=r1.test_rows,
            duration_seconds=r1.duration_seconds,
            tags=r1.tags
        )
        memory_tracker.log_run(r2)

        # Initially, no champion exists
        assert memory_tracker.get_champion() is None

        # Promote r1
        memory_tracker.set_champion("run_1")
        champ = memory_tracker.get_champion()
        assert champ is not None
        assert champ.run_id == "run_1"

        # Check in get_run that is_champion is indeed True
        assert memory_tracker.get_run("run_1").is_champion is True
        assert memory_tracker.get_run("run_2").is_champion is False

        # Promote r2 instead
        memory_tracker.set_champion("run_2")
        champ = memory_tracker.get_champion()
        assert champ is not None
        assert champ.run_id == "run_2"
        assert memory_tracker.get_run("run_1").is_champion is False
        assert memory_tracker.get_run("run_2").is_champion is True
