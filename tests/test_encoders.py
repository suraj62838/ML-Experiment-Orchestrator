"""
test_encoders.py — Unit tests for OneHotEncoderStep and LabelEncoderStep.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# ── Ensure project root is importable ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.steps.encoders import OneHotEncoderStep, LabelEncoderStep
from src.pipeline.base import NotFittedError


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def train_df():
    return pd.DataFrame({
        "gender": ["male", "female", "male", "other", "female"],
        "size":   ["S", "M", "L", "M", "S"],
        "score":  [1.0, 2.0, 3.0, 4.0, 5.0],
    })


@pytest.fixture
def test_df():
    return pd.DataFrame({
        "gender": ["female", "other", "male", "unknown"],  # "unknown" is unseen
        "size":   ["M", "L", "XL", "S"],                   # "XL" is unseen
        "score":  [1.5, 2.5, 3.5, 4.5],
    })


# ─────────────────────────────────────────────────────────────────────────────
# OneHotEncoderStep
# ─────────────────────────────────────────────────────────────────────────────

class TestOneHotEncoderStep:

    def test_fit_transform_produces_correct_columns(self, train_df):
        step = OneHotEncoderStep({"cols": ["gender"]})
        result = step.fit_transform(train_df)
        # Original 'gender' column should be gone
        assert "gender" not in result.columns
        # Indicator columns must exist
        assert "gender_male" in result.columns
        assert "gender_female" in result.columns
        assert "gender_other" in result.columns
        # Other columns untouched
        assert "score" in result.columns
        assert "size" in result.columns

    def test_indicator_values_are_correct(self, train_df):
        step = OneHotEncoderStep({"cols": ["gender"]})
        result = step.fit_transform(train_df)
        # Row 0 is "male"
        assert result["gender_male"].iloc[0] == 1
        assert result["gender_female"].iloc[0] == 0
        # Row 1 is "female"
        assert result["gender_female"].iloc[1] == 1
        assert result["gender_male"].iloc[1] == 0

    def test_unseen_category_is_ignored_not_error(self, train_df, test_df):
        """Transform on data with unseen category must NOT raise."""
        step = OneHotEncoderStep({"cols": ["gender"]})
        step.fit(train_df)
        result = step.transform(test_df)
        # "unknown" is unseen — no column for it
        assert "gender_unknown" not in result.columns
        # Row 3 has "unknown" gender → all known indicator columns are 0
        assert result["gender_male"].iloc[3] == 0
        assert result["gender_female"].iloc[3] == 0
        assert result["gender_other"].iloc[3] == 0

    def test_transform_without_fit_raises(self, train_df):
        step = OneHotEncoderStep({"cols": ["gender"]})
        with pytest.raises(NotFittedError):
            step.transform(train_df)

    def test_empty_cols_list_is_noop(self, train_df):
        step = OneHotEncoderStep({"cols": []})
        result = step.fit_transform(train_df)
        assert list(result.columns) == list(train_df.columns)

    def test_missing_column_warns_not_raises(self, train_df):
        step = OneHotEncoderStep({"cols": ["nonexistent"]})
        with pytest.warns(UserWarning, match="nonexistent"):
            step.fit(train_df)
        # transform should not crash either
        result = step.transform(train_df)
        assert list(result.columns) == list(train_df.columns)

    def test_multiple_cols_encoded(self, train_df):
        step = OneHotEncoderStep({"cols": ["gender", "size"]})
        result = step.fit_transform(train_df)
        assert "gender" not in result.columns
        assert "size" not in result.columns
        assert "gender_male" in result.columns
        assert "size_S" in result.columns

    def test_original_df_not_mutated(self, train_df):
        original_cols = list(train_df.columns)
        step = OneHotEncoderStep({"cols": ["gender"]})
        step.fit_transform(train_df)
        assert list(train_df.columns) == original_cols


# ─────────────────────────────────────────────────────────────────────────────
# LabelEncoderStep
# ─────────────────────────────────────────────────────────────────────────────

class TestLabelEncoderStep:

    def test_fit_transform_returns_integers(self, train_df):
        step = LabelEncoderStep({"cols": ["gender"]})
        result = step.fit_transform(train_df)
        assert result["gender"].dtype in (int, np.int64, np.int32)

    def test_mapping_is_stable(self, train_df):
        """Same value must always get the same integer label."""
        step = LabelEncoderStep({"cols": ["size"]})
        result = step.fit_transform(train_df)
        # Sorted order: L=0, M=1, S=2
        assert result["size"].iloc[0] == 2   # S
        assert result["size"].iloc[1] == 1   # M
        assert result["size"].iloc[2] == 0   # L

    def test_unseen_values_mapped_to_minus_one(self, train_df, test_df):
        step = LabelEncoderStep({"cols": ["size"]})
        step.fit(train_df)
        result = step.transform(test_df)
        # "XL" is unseen → -1
        xl_rows = test_df["size"] == "XL"
        assert (result.loc[xl_rows, "size"] == -1).all()

    def test_transform_without_fit_raises(self, train_df):
        step = LabelEncoderStep({"cols": ["gender"]})
        with pytest.raises(NotFittedError):
            step.transform(train_df)

    def test_fit_and_transform_are_separate(self, train_df, test_df):
        """Fitting on train must not process test data."""
        step = LabelEncoderStep({"cols": ["gender"]})
        step.fit(train_df)
        # Mapping was learned from train — transform uses it without refitting
        result = step.transform(test_df)
        assert "gender" in result.columns
        assert result["gender"].dtype in (int, np.int64, np.int32)

    def test_empty_cols_list_is_noop(self, train_df):
        step = LabelEncoderStep({"cols": []})
        result = step.fit_transform(train_df)
        pd.testing.assert_frame_equal(result, train_df)

    def test_missing_column_warns_not_raises(self, train_df):
        step = LabelEncoderStep({"cols": ["nonexistent"]})
        with pytest.warns(UserWarning, match="nonexistent"):
            step.fit(train_df)
        result = step.transform(train_df)
        assert list(result.columns) == list(train_df.columns)

    def test_original_df_not_mutated(self, train_df):
        original = train_df["gender"].tolist()
        step = LabelEncoderStep({"cols": ["gender"]})
        step.fit_transform(train_df)
        assert train_df["gender"].tolist() == original
