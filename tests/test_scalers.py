"""
test_scalers.py — Unit tests for StandardScalerStep and MinMaxScalerStep.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.steps.scalers import StandardScalerStep, MinMaxScalerStep
from src.pipeline.base import NotFittedError


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def train_df():
    np.random.seed(42)
    return pd.DataFrame({
        "a": np.random.normal(10, 2, 100),
        "b": np.random.normal(50, 10, 100),
        "cat": ["x"] * 100,        # non-numeric — should not be touched
    })


@pytest.fixture
def val_df():
    np.random.seed(99)
    return pd.DataFrame({
        "a": np.random.normal(10, 2, 30),
        "b": np.random.normal(50, 10, 30),
        "cat": ["y"] * 30,
    })


@pytest.fixture
def const_df():
    """DataFrame with constant column (zero std / zero range)."""
    return pd.DataFrame({"a": [5.0] * 50, "b": np.random.normal(0, 1, 50)})


# ─────────────────────────────────────────────────────────────────────────────
# StandardScalerStep
# ─────────────────────────────────────────────────────────────────────────────

class TestStandardScalerStep:

    def test_all_numeric_autodetects_correctly(self, train_df):
        step = StandardScalerStep({"cols": "all_numeric"})
        result = step.fit_transform(train_df)
        # Numeric cols should be scaled; "cat" unchanged
        assert "cat" in result.columns
        assert result["cat"].iloc[0] == "x"

    def test_mean_approx_zero_after_fit_transform(self, train_df):
        step = StandardScalerStep({"cols": "all_numeric"})
        result = step.fit_transform(train_df)
        assert abs(result["a"].mean()) < 1e-10
        assert abs(result["b"].mean()) < 1e-10

    def test_std_approx_one_after_fit_transform(self, train_df):
        step = StandardScalerStep({"cols": "all_numeric"})
        result = step.fit_transform(train_df)
        assert abs(result["a"].std(ddof=0) - 1.0) < 1e-10
        assert abs(result["b"].std(ddof=0) - 1.0) < 1e-10

    def test_val_uses_train_params_not_val_stats(self, train_df, val_df):
        """Val scaled with train mean/std — val mean won't be exactly 0."""
        step = StandardScalerStep({"cols": ["a", "b"]})
        step.fit(train_df)
        val_result = step.transform(val_df)
        # Val mean should be close to 0 but not exactly 0
        train_result = step.transform(train_df)
        assert abs(train_result["a"].mean()) < 1e-10
        # Verify val uses same mean as train (not re-computed)
        val_mean_a = val_df["a"].mean()
        train_mean_a = step._means["a"]
        expected_val_a_mean = (val_mean_a - train_mean_a) / step._stds["a"]
        assert abs(val_result["a"].mean() - expected_val_a_mean) < 1e-10

    def test_transform_without_fit_raises(self, train_df):
        step = StandardScalerStep({"cols": ["a"]})
        with pytest.raises(NotFittedError):
            step.transform(train_df)

    def test_zero_std_column_set_to_zero(self, const_df):
        step = StandardScalerStep({"cols": ["a"]})
        result = step.fit_transform(const_df)
        assert (result["a"] == 0.0).all()

    def test_explicit_col_list(self, train_df):
        """Only listed cols are scaled; others unchanged."""
        step = StandardScalerStep({"cols": ["a"]})
        result = step.fit_transform(train_df)
        assert abs(result["a"].mean()) < 1e-10
        # "b" was not in the config — should be original values
        pd.testing.assert_series_equal(result["b"], train_df["b"])

    def test_original_df_not_mutated(self, train_df):
        original_a = train_df["a"].copy()
        step = StandardScalerStep({"cols": "all_numeric"})
        step.fit_transform(train_df)
        pd.testing.assert_series_equal(train_df["a"], original_a)


# ─────────────────────────────────────────────────────────────────────────────
# MinMaxScalerStep
# ─────────────────────────────────────────────────────────────────────────────

class TestMinMaxScalerStep:

    def test_output_range_is_zero_to_one_on_train(self, train_df):
        step = MinMaxScalerStep({"cols": ["a", "b"]})
        result = step.fit_transform(train_df)
        assert result["a"].min() >= 0.0 - 1e-10
        assert result["a"].max() <= 1.0 + 1e-10
        assert result["b"].min() >= 0.0 - 1e-10
        assert result["b"].max() <= 1.0 + 1e-10

    def test_min_is_zero_max_is_one_exactly_on_train(self, train_df):
        step = MinMaxScalerStep({"cols": ["a"]})
        result = step.fit_transform(train_df)
        assert abs(result["a"].min()) < 1e-10
        assert abs(result["a"].max() - 1.0) < 1e-10

    def test_val_uses_train_min_max(self, train_df, val_df):
        """Val uses train min/max — values may exceed [0, 1]."""
        step = MinMaxScalerStep({"cols": ["a", "b"]})
        step.fit(train_df)
        val_result = step.transform(val_df)
        # Verify formula: (val_a - train_min) / (train_max - train_min)
        train_min = step._mins["a"]
        train_max = step._maxs["a"]
        expected = (val_df["a"] - train_min) / (train_max - train_min)
        pd.testing.assert_series_equal(
            val_result["a"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_transform_without_fit_raises(self, train_df):
        step = MinMaxScalerStep({"cols": ["a"]})
        with pytest.raises(NotFittedError):
            step.transform(train_df)

    def test_zero_range_column_set_to_zero(self, const_df):
        step = MinMaxScalerStep({"cols": ["a"]})
        result = step.fit_transform(const_df)
        assert (result["a"] == 0.0).all()

    def test_all_numeric_autodetects(self, train_df):
        step = MinMaxScalerStep({"cols": "all_numeric"})
        result = step.fit_transform(train_df)
        assert "cat" in result.columns
        assert result["cat"].iloc[0] == "x"
        assert result["a"].min() >= 0.0 - 1e-10

    def test_original_df_not_mutated(self, train_df):
        original_a = train_df["a"].copy()
        step = MinMaxScalerStep({"cols": ["a", "b"]})
        step.fit_transform(train_df)
        pd.testing.assert_series_equal(train_df["a"], original_a)
