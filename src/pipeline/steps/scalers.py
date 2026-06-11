"""
scalers.py — Numeric feature scaling pipeline steps.

Steps
-----
StandardScalerStep
    Standardises features to zero mean and unit variance using statistics
    computed on training data only.  Formula: ``(x - mean) / std``.

MinMaxScalerStep
    Scales features to the ``[0, 1]`` range using min/max computed on
    training data only.  Formula: ``(x - min) / (max - min)``.

Both steps accept ``cols="all_numeric"`` to automatically detect and
scale all numeric columns present at fit time.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.pipeline.base import PipelineStep
from src.utils.logger import get_logger

logger = get_logger(__name__)

_ALL_NUMERIC = "all_numeric"


def _resolve_cols(df: pd.DataFrame, cols: list[str] | str) -> list[str]:
    """Return the list of columns to operate on.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame being processed (used for auto-detection).
    cols : list[str] or ``"all_numeric"``
        Either an explicit list of column names or the sentinel string
        ``"all_numeric"`` to auto-detect all numeric columns.

    Returns
    -------
    list[str]
        Resolved column names present in *df*.
    """
    if cols == _ALL_NUMERIC:
        return df.select_dtypes(include="number").columns.tolist()
    return [c for c in cols if c in df.columns]


class StandardScalerStep(PipelineStep):
    """Standardise numeric columns to zero mean and unit variance.

    Computes mean and standard deviation from training data during ``fit``.
    Applies ``(x - mean) / std`` during ``transform``.  Columns with zero
    standard deviation are left unchanged (division by zero avoided).

    Parameters
    ----------
    config : dict
        ``cols`` : list[str] or ``"all_numeric"``
            Columns to scale.  Use ``"all_numeric"`` to auto-detect.

    Examples
    --------
    >>> step = StandardScalerStep({"cols": "all_numeric"})
    >>> X_train_scaled = step.fit_transform(X_train)
    >>> X_test_scaled  = step.transform(X_test)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._cols_cfg: list[str] | str = config.get("cols", _ALL_NUMERIC)
        # Fitted parameters (col -> value)
        self._means: dict[str, float] = {}
        self._stds: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "StandardScalerStep":
        """Compute mean and std for each target column from training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training data.

        Returns
        -------
        StandardScalerStep
            Returns ``self`` for chaining.
        """
        cols = _resolve_cols(df, self._cols_cfg)
        for col in cols:
            self._means[col] = float(df[col].mean())
            self._stds[col] = float(df[col].std(ddof=0))
            logger.debug(
                "StandardScalerStep fit: col='%s' mean=%.4f std=%.4f",
                col,
                self._means[col],
                self._stds[col],
            )

        self._fitted = True
        logger.info("StandardScalerStep fitted on %d column(s): %s", len(cols), cols)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply standardisation using fitted mean and std.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with scaled numeric columns.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called yet.
        """
        self._check_fitted()
        df = df.copy()

        for col, mean in self._means.items():
            if col not in df.columns:
                logger.warning(
                    "StandardScalerStep transform: column '%s' not found — skipping.", col
                )
                continue
            std = self._stds[col]
            if std == 0.0:
                logger.warning(
                    "StandardScalerStep: col='%s' has zero std — column left unchanged.", col
                )
                df[col] = 0.0
            else:
                df[col] = (df[col] - mean) / std

        return df


class MinMaxScalerStep(PipelineStep):
    """Scale numeric columns to the ``[0, 1]`` range.

    Computes min and max from training data during ``fit``.  Applies
    ``(x - min) / (max - min)`` during ``transform``.  Columns where
    ``max == min`` are set to ``0.0`` to avoid division by zero.

    Parameters
    ----------
    config : dict
        ``cols`` : list[str] or ``"all_numeric"``
            Columns to scale.  Use ``"all_numeric"`` to auto-detect.

    Notes
    -----
    Validation/test values outside the training range will produce outputs
    outside ``[0, 1]`` — this is expected behaviour (no clipping applied).

    Examples
    --------
    >>> step = MinMaxScalerStep({"cols": ["age", "income"]})
    >>> X_train_scaled = step.fit_transform(X_train)
    >>> X_val_scaled   = step.transform(X_val)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._cols_cfg: list[str] | str = config.get("cols", _ALL_NUMERIC)
        # Fitted parameters
        self._mins: dict[str, float] = {}
        self._maxs: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "MinMaxScalerStep":
        """Compute min and max for each target column from training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training data.

        Returns
        -------
        MinMaxScalerStep
            Returns ``self`` for chaining.
        """
        cols = _resolve_cols(df, self._cols_cfg)
        for col in cols:
            self._mins[col] = float(df[col].min())
            self._maxs[col] = float(df[col].max())
            logger.debug(
                "MinMaxScalerStep fit: col='%s' min=%.4f max=%.4f",
                col,
                self._mins[col],
                self._maxs[col],
            )

        self._fitted = True
        logger.info("MinMaxScalerStep fitted on %d column(s): %s", len(cols), cols)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply min-max scaling using fitted parameters.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with scaled numeric columns.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called yet.
        """
        self._check_fitted()
        df = df.copy()

        for col, min_val in self._mins.items():
            if col not in df.columns:
                logger.warning(
                    "MinMaxScalerStep transform: column '%s' not found — skipping.", col
                )
                continue
            max_val = self._maxs[col]
            rng = max_val - min_val
            if rng == 0.0:
                logger.warning(
                    "MinMaxScalerStep: col='%s' has zero range — column set to 0.0.", col
                )
                df[col] = 0.0
            else:
                df[col] = (df[col] - min_val) / rng

        return df
