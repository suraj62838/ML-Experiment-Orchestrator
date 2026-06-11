"""
feature_ops.py — Feature engineering pipeline steps.

Steps
-----
LogTransformStep
    Applies log1p (natural log of 1+x) to the specified columns.
    Warns at fit time if any training values are negative.

PolynomialFeaturesStep
    Adds polynomial and pairwise interaction columns.
    New columns are named clearly: "age^2", "age*income", etc.
"""

from __future__ import annotations
import warnings
from typing import Any
import numpy as np
import pandas as pd
from src.pipeline.base import PipelineStep
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LogTransformStep(PipelineStep):
    """Apply log1p transformation to specified numeric columns.

    Uses ``numpy.log1p`` (i.e. ``log(1 + x)``) which is numerically stable
    for values near zero.  During ``fit`` the step validates that all
    training values are non-negative and warns if any negative values are
    found (the transform will still proceed, but results for negative values
    will be ``NaN``).

    Parameters
    ----------
    config : dict
        ``cols`` : list[str]  — Column names to transform.

    Examples
    --------
    >>> step = LogTransformStep({"cols": ["salary", "revenue"]})
    >>> X_train_log = step.fit_transform(X_train)
    >>> X_val_log   = step.transform(X_val)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._cols: list[str] = list(config.get("cols", []))

    def fit(self, df: pd.DataFrame) -> "LogTransformStep":
        """Validate that training values are non-negative.

        Parameters
        ----------
        df : pd.DataFrame
            Training data.

        Returns
        -------
        LogTransformStep
            Returns ``self`` for chaining.
        """
        present = [c for c in self._cols if c in df.columns]
        missing = set(self._cols) - set(present)
        if missing:
            warnings.warn(
                f"LogTransformStep: columns not found: {sorted(missing)}",
                UserWarning, stacklevel=2,
            )
        for col in present:
            n_neg = int((df[col] < 0).sum())
            if n_neg:
                warnings.warn(
                    f"LogTransformStep: column '{col}' contains {n_neg} negative "
                    "value(s). log1p of negative values produces NaN.",
                    UserWarning, stacklevel=2,
                )
        self._fitted = True
        logger.info("LogTransformStep fitted on column(s): %s", present)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log1p to configured columns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with log1p-transformed columns.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called.
        """
        self._check_fitted()
        df = df.copy()
        for col in self._cols:
            if col not in df.columns:
                logger.warning("LogTransformStep: column '%s' not found — skipping.", col)
                continue
            df[col] = np.log1p(df[col])
            logger.debug("LogTransformStep: applied log1p to '%s'.", col)
        return df


class PolynomialFeaturesStep(PipelineStep):
    """Add polynomial and pairwise interaction features.

    For each column in ``cols`` adds ``col^2``, ``col^3``, … up to ``degree``.
    For each pair of columns ``(a, b)`` adds ``a*b`` interaction columns.

    New column names follow the conventions:
    - ``"age^2"`` for ``age`` squared
    - ``"age*income"`` for the product of ``age`` and ``income``

    Parameters
    ----------
    config : dict
        ``cols``   : list[str] — Source columns.
        ``degree`` : int, default ``2`` — Maximum polynomial degree.

    Notes
    -----
    * The original columns are **kept** in the output alongside the new ones.
    * Fit simply stores the configuration (no data-dependent statistics).

    Examples
    --------
    >>> step = PolynomialFeaturesStep({"cols": ["age", "income"], "degree": 2})
    >>> X_poly = step.fit_transform(X_train)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._cols: list[str] = list(config.get("cols", []))
        self._degree: int = int(config.get("degree", 2))

    def fit(self, df: pd.DataFrame) -> "PolynomialFeaturesStep":
        """Store configuration (no data-dependent computation needed).

        Parameters
        ----------
        df : pd.DataFrame
            Training data (used only to warn about missing columns).

        Returns
        -------
        PolynomialFeaturesStep
            Returns ``self`` for chaining.
        """
        missing = [c for c in self._cols if c not in df.columns]
        if missing:
            warnings.warn(
                f"PolynomialFeaturesStep: columns not found: {missing}",
                UserWarning, stacklevel=2,
            )
        self._fitted = True
        logger.info(
            "PolynomialFeaturesStep fitted — cols=%s degree=%d", self._cols, self._degree
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add polynomial and interaction columns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with additional feature columns appended.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called.
        """
        self._check_fitted()
        df = df.copy()
        present = [c for c in self._cols if c in df.columns]

        # Polynomial terms: col^2, col^3, ...
        for col in present:
            for deg in range(2, self._degree + 1):
                new_col = f"{col}^{deg}"
                df[new_col] = df[col] ** deg
                logger.debug("PolynomialFeaturesStep: added '%s'.", new_col)

        # Pairwise interaction terms: a*b
        for i, col_a in enumerate(present):
            for col_b in present[i + 1:]:
                new_col = f"{col_a}*{col_b}"
                df[new_col] = df[col_a] * df[col_b]
                logger.debug("PolynomialFeaturesStep: added '%s'.", new_col)

        n_new = len(df.columns) - len(df.columns) + (
            len(present) * (self._degree - 1) +
            len(present) * (len(present) - 1) // 2
        )
        logger.info(
            "PolynomialFeaturesStep: added %d new feature column(s).", n_new
        )
        return df
