"""
imputers.py — Missing-value imputation pipeline steps.

Steps: MeanImputerStep, MedianImputerStep, ModeImputerStep.
All accept cols="all_numeric" or "all_categorical" for auto-detection.
"""

from __future__ import annotations
from typing import Any
import pandas as pd
from src.pipeline.base import PipelineStep
from src.utils.logger import get_logger

logger = get_logger(__name__)

_ALL_NUMERIC = "all_numeric"
_ALL_CATEGORICAL = "all_categorical"


def _resolve_imputer_cols(df: pd.DataFrame, cols) -> list[str]:
    if cols == _ALL_NUMERIC:
        return df.select_dtypes(include="number").columns.tolist()
    if cols == _ALL_CATEGORICAL:
        return df.select_dtypes(include=["object", "category"]).columns.tolist()
    return [c for c in cols if c in df.columns]


class MeanImputerStep(PipelineStep):
    """Impute missing values in numeric columns with the column mean.

    Fit computes mean from training data only; transform applies it.
    Accepts cols="all_numeric" to auto-detect numeric columns at fit time.

    Parameters
    ----------
    config : dict
        ``cols`` : list[str] or ``"all_numeric"``
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._cols_cfg = config.get("cols", _ALL_NUMERIC)
        self._fill_values: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "MeanImputerStep":
        """Compute column means from training data."""
        cols = _resolve_imputer_cols(df, self._cols_cfg)
        for col in cols:
            self._fill_values[col] = float(df[col].mean())
        self._fitted = True
        logger.info("MeanImputerStep fitted on %d column(s): %s", len(cols), cols)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill nulls with fitted column means.

        Raises
        ------
        NotFittedError
            If fit has not been called.
        """
        self._check_fitted()
        df = df.copy()
        for col, fill_val in self._fill_values.items():
            if col not in df.columns:
                logger.warning("MeanImputerStep: column '%s' not found — skipping.", col)
                continue
            n = int(df[col].isna().sum())
            if n:
                df[col] = df[col].fillna(fill_val)
                logger.info("MeanImputerStep: filled %d null(s) in '%s' (mean=%.4f)", n, col, fill_val)
        return df


class MedianImputerStep(PipelineStep):
    """Impute missing values in numeric columns with the column median.

    More robust to outliers than mean. Fit computes median from training
    data only. Accepts cols="all_numeric".

    Parameters
    ----------
    config : dict
        ``cols`` : list[str] or ``"all_numeric"``
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._cols_cfg = config.get("cols", _ALL_NUMERIC)
        self._fill_values: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "MedianImputerStep":
        """Compute column medians from training data."""
        cols = _resolve_imputer_cols(df, self._cols_cfg)
        for col in cols:
            self._fill_values[col] = float(df[col].median())
        self._fitted = True
        logger.info("MedianImputerStep fitted on %d column(s): %s", len(cols), cols)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill nulls with fitted column medians.

        Raises
        ------
        NotFittedError
            If fit has not been called.
        """
        self._check_fitted()
        df = df.copy()
        for col, fill_val in self._fill_values.items():
            if col not in df.columns:
                logger.warning("MedianImputerStep: column '%s' not found — skipping.", col)
                continue
            n = int(df[col].isna().sum())
            if n:
                df[col] = df[col].fillna(fill_val)
                logger.info("MedianImputerStep: filled %d null(s) in '%s' (median=%.4f)", n, col, fill_val)
        return df


class ModeImputerStep(PipelineStep):
    """Impute missing values with the column mode (most frequent value).

    Works on numeric and categorical columns. Fit computes mode from
    training data only. Accepts cols="all_categorical" or "all_numeric".

    Parameters
    ----------
    config : dict
        ``cols`` : list[str], ``"all_numeric"``, or ``"all_categorical"``
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._cols_cfg = config.get("cols", _ALL_CATEGORICAL)
        self._fill_values: dict[str, Any] = {}

    def fit(self, df: pd.DataFrame) -> "ModeImputerStep":
        """Compute column modes from training data."""
        cols = _resolve_imputer_cols(df, self._cols_cfg)
        for col in cols:
            mode_s = df[col].mode()
            if mode_s.empty:
                logger.warning("ModeImputerStep: col='%s' is all-null — skipping.", col)
                continue
            self._fill_values[col] = mode_s.iloc[0]
        self._fitted = True
        logger.info("ModeImputerStep fitted on %d column(s): %s", len(cols), cols)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill nulls with fitted column modes.

        Raises
        ------
        NotFittedError
            If fit has not been called.
        """
        self._check_fitted()
        df = df.copy()
        for col, fill_val in self._fill_values.items():
            if col not in df.columns:
                logger.warning("ModeImputerStep: column '%s' not found — skipping.", col)
                continue
            n = int(df[col].isna().sum())
            if n:
                df[col] = df[col].fillna(fill_val)
                logger.info("ModeImputerStep: filled %d null(s) in '%s' (mode=%r)", n, col, fill_val)
        return df
