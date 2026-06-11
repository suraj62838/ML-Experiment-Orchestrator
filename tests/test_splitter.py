"""
test_splitter.py — Unit tests for DataSplitter.

All tests use small in-memory DataFrames.
Run with: pytest tests/
"""

from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.splitter import DataSplitter


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """A 100-row DataFrame large enough for stratified splitting."""
    import numpy as np
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "feat_a": rng.standard_normal(100),
            "feat_b": rng.standard_normal(100),
            "target": rng.integers(0, 2, size=100),
        }
    )


@pytest.fixture()
def default_splitter() -> DataSplitter:
    """A DataSplitter with standard configuration."""
    return DataSplitter(test_size=0.2, val_size=0.1, random_seed=42, stratify=False)


# ─────────────────────────────────────────────────────────────────────────────
# Tests — output keys
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitKeys:
    """Verify the dict returned by split() contains all expected keys."""

    def test_returns_all_six_keys(self, sample_df, default_splitter):
        """split() must return exactly the six expected keys."""
        result = default_splitter.split(sample_df, "target")
        expected_keys = {"X_train", "X_val", "X_test", "y_train", "y_val", "y_test"}
        assert set(result.keys()) == expected_keys

    def test_features_dont_contain_target(self, sample_df, default_splitter):
        """Feature matrices must not contain the target column."""
        result = default_splitter.split(sample_df, "target")
        for key in ("X_train", "X_val", "X_test"):
            assert "target" not in result[key].columns


# ─────────────────────────────────────────────────────────────────────────────
# Tests — sizes
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitSizes:
    """Verify that partition sizes are within acceptable bounds."""

    def test_all_rows_accounted_for(self, sample_df, default_splitter):
        """Total rows across train + val + test must equal original row count."""
        result = default_splitter.split(sample_df, "target")
        total = len(result["X_train"]) + len(result["X_val"]) + len(result["X_test"])
        assert total == len(sample_df)

    def test_test_size_is_approximately_correct(self, sample_df, default_splitter):
        """Test set should be roughly 20 % of the dataset."""
        result = default_splitter.split(sample_df, "target")
        test_fraction = len(result["X_test"]) / len(sample_df)
        assert 0.15 <= test_fraction <= 0.25

    def test_x_and_y_sizes_match(self, sample_df, default_splitter):
        """Feature matrix and target vector must be the same length for every split."""
        result = default_splitter.split(sample_df, "target")
        assert len(result["X_train"]) == len(result["y_train"])
        assert len(result["X_val"]) == len(result["y_val"])
        assert len(result["X_test"]) == len(result["y_test"])


# ─────────────────────────────────────────────────────────────────────────────
# Tests — reproducibility
# ─────────────────────────────────────────────────────────────────────────────

class TestReproducibility:
    """Verify that the same seed always produces identical splits."""

    def test_same_seed_same_indices(self, sample_df):
        """Two splitters with the same random_seed must yield identical index sets."""
        s1 = DataSplitter(test_size=0.2, val_size=0.1, random_seed=99, stratify=False)
        s2 = DataSplitter(test_size=0.2, val_size=0.1, random_seed=99, stratify=False)

        r1 = s1.split(sample_df, "target")
        r2 = s2.split(sample_df, "target")

        for key in ("X_train", "X_val", "X_test"):
            assert list(r1[key].index) == list(r2[key].index)

    def test_different_seeds_different_indices(self, sample_df):
        """Different seeds should (very likely) produce different test-set indices."""
        s1 = DataSplitter(test_size=0.2, val_size=0.1, random_seed=1, stratify=False)
        s2 = DataSplitter(test_size=0.2, val_size=0.1, random_seed=2, stratify=False)

        r1 = s1.split(sample_df, "target")
        r2 = s2.split(sample_df, "target")

        # It is astronomically unlikely for both seeds to produce the same order
        assert list(r1["X_test"].index) != list(r2["X_test"].index)


# ─────────────────────────────────────────────────────────────────────────────
# Tests — error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitErrors:
    """Error handling tests for DataSplitter."""

    def test_raises_on_missing_target_col(self, sample_df, default_splitter):
        """A KeyError is raised when the target column is absent from the DataFrame."""
        with pytest.raises(KeyError, match="not found"):
            default_splitter.split(sample_df, "nonexistent_col")
