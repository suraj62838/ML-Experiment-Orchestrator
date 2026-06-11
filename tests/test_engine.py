"""
test_engine.py — Unit tests for PipelineEngine.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.engine import PipelineEngine
from src.pipeline.base import NotFittedError


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def train_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "gender": ["male", "female", "male", "female", "male"],
        "age": [25.0, 30.0, np.nan, 45.0, 50.0],
        "score": [10.0, 20.0, 30.0, 40.0, np.nan],
    })


@pytest.fixture
def test_df():
    return pd.DataFrame({
        "id": [6, 7],
        "gender": ["female", "other"],  # "other" is unseen in train
        "age": [np.nan, 35.0],
        "score": [15.0, np.nan],
    })


# ─────────────────────────────────────────────────────────────────────────────
# PipelineEngine tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineEngine:

    def test_pipeline_runs_in_correct_order_and_transforms(self, train_df):
        # Steps sequence:
        # 1. Drop "id"
        # 2. Mean impute age and score
        # 3. One-hot encode gender
        # 4. Standard scale all numeric (which now includes the one-hot columns if they are numeric)
        steps_config = [
            {"type": "drop_columns", "cols": ["id"]},
            {"type": "mean_impute", "cols": "all_numeric"},
            {"type": "onehot_encode", "cols": ["gender"]},
            {"type": "standard_scale", "cols": "all_numeric"},
        ]
        
        engine = PipelineEngine(steps_config)
        
        # Fit & Transform
        transformed = engine.fit_transform(train_df)
        
        # 1. "id" and "gender" should be dropped/encoded
        assert "id" not in transformed.columns
        assert "gender" not in transformed.columns
        
        # 2. One-hot columns should exist
        assert "gender_male" in transformed.columns
        assert "gender_female" in transformed.columns
        
        # 3. Impuations: no null values should be left
        assert transformed.isnull().sum().sum() == 0
        
        # 4. StandardScaler: columns should be standardized (mean close to 0, std close to 1)
        # We check some of them:
        assert abs(transformed["age"].mean()) < 1e-10
        assert abs(transformed["score"].mean()) < 1e-10
        
    def test_fit_on_train_does_not_touch_test_data(self, train_df, test_df):
        steps_config = [
            {"type": "mean_impute", "cols": ["age"]},
            {"type": "standard_scale", "cols": ["age"]},
        ]
        engine = PipelineEngine(steps_config)
        
        # Fit on train_df
        engine.fit(train_df)
        
        # Check that we learned the mean from train_df
        # Mean of [25, 30, 45, 50] is 37.5
        imputer_step = engine._steps[0]
        scaler_step = engine._steps[1]
        
        assert imputer_step._fill_values["age"] == 37.5
        assert scaler_step._means["age"] == 37.5  # Mean of age after imputation (all values are 37.5 now)
        
        # Transform test_df
        transformed_test = engine.transform(test_df)
        
        # In test_df: first element of 'age' is NaN.
        # It should be imputed with 37.5 (from train_df), and then scaled.
        # Scaled value for 37.5 should be 0.0 because it equals the scaler mean of 37.5.
        assert abs(transformed_test["age"].iloc[0]) < 1e-10
        
    def test_unknown_step_type_raises_clear_error(self):
        steps_config = [
            {"type": "invalid_step_name", "cols": ["age"]},
        ]
        engine = PipelineEngine(steps_config)
        
        # Try to fit should raise ValueError containing available steps
        with pytest.raises(ValueError) as excinfo:
            engine.fit(pd.DataFrame({"age": [1, 2, 3]}))
            
        assert "Unknown pipeline step type: 'invalid_step_name'" in str(excinfo.value)
        assert "Available steps" in str(excinfo.value)
        
    def test_get_steps_summary_returns_correct_structure(self, train_df):
        steps_config = [
            {"type": "drop_columns", "cols": ["id"]},
            {"type": "mean_impute", "cols": ["age"]},
        ]
        engine = PipelineEngine(steps_config)
        engine.fit(train_df)
        
        summary = engine.get_steps_summary()
        assert len(summary) == 2
        assert summary[0]["name"] == "drop_columns"
        assert "cols" in summary[0]["config"]
        assert summary[1]["name"] == "mean_impute"
        assert "cols" in summary[1]["config"]

    def test_transform_before_fit_raises_error(self, train_df):
        steps_config = [
            {"type": "drop_columns", "cols": ["id"]},
        ]
        engine = PipelineEngine(steps_config)
        with pytest.raises(RuntimeError) as excinfo:
            engine.transform(train_df)
        assert "PipelineEngine has not been fitted" in str(excinfo.value)
