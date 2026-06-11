"""
test_cleaner.py — Unit tests for DataCleaner.

All tests use small in-memory DataFrames.
Run with: pytest tests/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cleaner import DataCleaner


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def df_with_nulls() -> pd.DataFrame:
    """DataFrame containing known null values for fill tests."""
    return pd.DataFrame(
        {"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan], "target": [0, 1, 0]}
    )


@pytest.fixture()
def df_with_duplicates() -> pd.DataFrame:
    """DataFrame containing one duplicate row."""
    return pd.DataFrame(
        {"a": [1, 2, 1], "b": [10, 20, 10], "target": [0, 1, 0]}
    )


@pytest.fixture()
def df_clean() -> pd.DataFrame:
    """Fully clean DataFrame with no nulls or duplicates."""
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "target": [0, 1, 0]})


# ─────────────────────────────────────────────────────────────────────────────
# Tests — null filling
# ─────────────────────────────────────────────────────────────────────────────

class TestFillNulls:
    """Tests for different fill_nulls strategies."""

    def test_fill_mean_removes_nulls(self, df_with_nulls):
        """After fill_nulls='mean', no NaN values should remain in numeric cols."""
        cleaner = DataCleaner({"fill_nulls": "mean", "drop_duplicates": False, "fix_dtypes": False})
        result = cleaner.clean(df_with_nulls)
        assert result[["a", "b"]].isnull().sum().sum() == 0

    def test_fill_median_removes_nulls(self, df_with_nulls):
        """fill_nulls='median' should fill all numeric NaNs."""
        cleaner = DataCleaner({"fill_nulls": "median", "drop_duplicates": False, "fix_dtypes": False})
        result = cleaner.clean(df_with_nulls)
        assert result[["a", "b"]].isnull().sum().sum() == 0

    def test_fill_mode_removes_nulls(self, df_with_nulls):
        """fill_nulls='mode' should fill all NaNs using the modal value."""
        cleaner = DataCleaner({"fill_nulls": "mode", "drop_duplicates": False, "fix_dtypes": False})
        result = cleaner.clean(df_with_nulls)
        assert result.isnull().sum().sum() == 0

    def test_fill_drop_removes_rows(self, df_with_nulls):
        """fill_nulls='drop' should remove all rows with any NaN."""
        cleaner = DataCleaner({"fill_nulls": "drop", "drop_duplicates": False, "fix_dtypes": False})
        result = cleaner.clean(df_with_nulls)
        assert result.isnull().sum().sum() == 0
        assert len(result) < len(df_with_nulls)

    def test_mean_fill_value_is_correct(self):
        """The filled value should equal the column mean of the original data."""
        df = pd.DataFrame({"a": [2.0, 4.0, np.nan], "b": [1, 2, 3]})
        cleaner = DataCleaner({"fill_nulls": "mean", "drop_duplicates": False, "fix_dtypes": False})
        result = cleaner.clean(df)
        # mean of [2.0, 4.0] = 3.0
        assert result["a"].iloc[2] == pytest.approx(3.0)

    def test_invalid_strategy_raises(self):
        """An unrecognised fill_nulls value should raise ValueError at init."""
        with pytest.raises(ValueError, match="Unknown fill_nulls strategy"):
            DataCleaner({"fill_nulls": "interpolate"})


# ─────────────────────────────────────────────────────────────────────────────
# Tests — drop duplicates
# ─────────────────────────────────────────────────────────────────────────────

class TestDropDuplicates:
    """Tests for the drop_duplicates option."""

    def test_duplicates_are_removed(self, df_with_duplicates):
        """Duplicate rows must be removed when drop_duplicates=True."""
        cleaner = DataCleaner({"fill_nulls": "mean", "drop_duplicates": True, "fix_dtypes": False})
        result = cleaner.clean(df_with_duplicates)
        assert len(result) == 2

    def test_duplicates_kept_when_disabled(self, df_with_duplicates):
        """Duplicate rows must be kept when drop_duplicates=False."""
        cleaner = DataCleaner({"fill_nulls": "mean", "drop_duplicates": False, "fix_dtypes": False})
        result = cleaner.clean(df_with_duplicates)
        assert len(result) == len(df_with_duplicates)


# ─────────────────────────────────────────────────────────────────────────────
# Tests — dtype fixing
# ─────────────────────────────────────────────────────────────────────────────

class TestFixDtypes:
    """Tests for the fix_dtypes coercion option."""

    def test_numeric_strings_are_coerced(self):
        """Object columns containing numeric strings should become numeric."""
        df = pd.DataFrame({"a": ["1", "2", "3"], "target": [0, 1, 0]})
        cleaner = DataCleaner({"fill_nulls": "mean", "drop_duplicates": False, "fix_dtypes": True})
        result = cleaner.clean(df)
        assert pd.api.types.is_numeric_dtype(result["a"])

    def test_non_numeric_strings_are_untouched(self):
        """Object columns with non-numeric text should remain as object dtype."""
        df = pd.DataFrame({"label": ["cat", "dog", "cat"], "target": [0, 1, 0]})
        cleaner = DataCleaner({"fill_nulls": "mean", "drop_duplicates": False, "fix_dtypes": True})
        result = cleaner.clean(df)
        assert result["label"].dtype == object


# ─────────────────────────────────────────────────────────────────────────────
# Tests — immutability
# ─────────────────────────────────────────────────────────────────────────────

class TestImmutability:
    """Ensure the original DataFrame is never mutated."""

    def test_original_df_not_modified(self, df_with_nulls):
        """clean() must not modify the input DataFrame in place."""
        original_nulls = df_with_nulls.isnull().sum().sum()
        cleaner = DataCleaner({"fill_nulls": "mean", "drop_duplicates": True, "fix_dtypes": True})
        cleaner.clean(df_with_nulls)
        assert df_with_nulls.isnull().sum().sum() == original_nulls
