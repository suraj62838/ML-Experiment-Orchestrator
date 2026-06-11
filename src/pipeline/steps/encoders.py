"""
encoders.py — Categorical encoding pipeline steps.

Steps
-----
OneHotEncoderStep
    Converts categorical columns to one-hot (binary indicator) columns.
    Unknown categories seen at transform time are silently ignored, so the
    step never raises on unseen data.

LabelEncoderStep
    Converts each categorical column to integer codes based on the mapping
    learned during ``fit``.  Unseen categories are mapped to ``-1``.
"""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd

from src.pipeline.base import PipelineStep
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OneHotEncoderStep(PipelineStep):
    """One-hot encode a list of categorical columns.

    For each column listed in ``config["cols"]`` the step learns the set of
    categories present in the training data and creates one binary indicator
    column per category.  The original columns are dropped from the output.

    At transform time, any category not seen during ``fit`` is silently
    ignored (``handle_unknown="ignore"`` semantics), preventing errors on
    validation or test data.

    Parameters
    ----------
    config : dict
        ``cols`` : list[str]
            Column names to one-hot encode.

    Examples
    --------
    >>> step = OneHotEncoderStep({"cols": ["gender"]})
    >>> X_train_enc = step.fit_transform(X_train)
    >>> X_test_enc  = step.transform(X_test)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._cols: list[str] = list(config.get("cols", []))
        # Mapping col -> list of categories (in sorted order, for stability)
        self._categories: dict[str, list] = {}

    def fit(self, df: pd.DataFrame) -> "OneHotEncoderStep":
        """Learn the category set for each configured column.

        Parameters
        ----------
        df : pd.DataFrame
            Training data.  Only the columns in ``config["cols"]`` are examined.

        Returns
        -------
        OneHotEncoderStep
            Returns ``self`` for chaining.
        """
        present = [c for c in self._cols if c in df.columns]
        missing = set(self._cols) - set(present)
        if missing:
            warnings.warn(
                f"OneHotEncoderStep: columns not found in DataFrame and will be "
                f"skipped: {sorted(missing)}",
                UserWarning,
                stacklevel=2,
            )

        for col in present:
            cats = sorted(df[col].dropna().unique().tolist())
            self._categories[col] = cats
            logger.debug("OneHotEncoderStep fit: col='%s' categories=%s", col, cats)

        self._fitted = True
        logger.info(
            "OneHotEncoderStep fitted on %d column(s): %s", len(present), present
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply one-hot encoding using the fitted category sets.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to transform (may be train, val, or test).

        Returns
        -------
        pd.DataFrame
            Copy of *df* with original columns replaced by indicator columns.
            New column names follow the pattern ``"<col>_<category>"``.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called yet.
        """
        self._check_fitted()
        df = df.copy()

        for col, categories in self._categories.items():
            if col not in df.columns:
                logger.warning(
                    "OneHotEncoderStep transform: column '%s' not found — skipping.", col
                )
                continue

            # Build indicator columns only for known categories
            for cat in categories:
                new_col = f"{col}_{cat}"
                df[new_col] = (df[col] == cat).astype(int)

            df = df.drop(columns=[col])
            logger.debug(
                "OneHotEncoderStep: encoded '%s' into %d indicator column(s).",
                col,
                len(categories),
            )

        return df


class LabelEncoderStep(PipelineStep):
    """Encode categorical columns as integer labels.

    Each unique value in a column is assigned an integer in the range
    ``[0, n_categories - 1]`` based on the sorted order of categories seen
    during ``fit``.  Values not seen during ``fit`` are mapped to ``-1``.

    Parameters
    ----------
    config : dict
        ``cols`` : list[str]
            Column names to label-encode.

    Examples
    --------
    >>> step = LabelEncoderStep({"cols": ["size"]})
    >>> X_train_enc = step.fit_transform(X_train)
    >>> X_test_enc  = step.transform(X_test)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._cols: list[str] = list(config.get("cols", []))
        # Mapping col -> {value: int_label}
        self._mappings: dict[str, dict] = {}

    def fit(self, df: pd.DataFrame) -> "LabelEncoderStep":
        """Build the label mapping for each configured column.

        Parameters
        ----------
        df : pd.DataFrame
            Training data.

        Returns
        -------
        LabelEncoderStep
            Returns ``self`` for chaining.
        """
        present = [c for c in self._cols if c in df.columns]
        missing = set(self._cols) - set(present)
        if missing:
            warnings.warn(
                f"LabelEncoderStep: columns not found in DataFrame and will be "
                f"skipped: {sorted(missing)}",
                UserWarning,
                stacklevel=2,
            )

        for col in present:
            unique_vals = sorted(df[col].dropna().unique().tolist(), key=str)
            self._mappings[col] = {val: idx for idx, val in enumerate(unique_vals)}
            logger.debug("LabelEncoderStep fit: col='%s' mapping=%s", col, self._mappings[col])

        self._fitted = True
        logger.info(
            "LabelEncoderStep fitted on %d column(s): %s", len(present), present
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted label mapping to each configured column.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with encoded integer columns (dtype ``int64``).
            Unseen values are mapped to ``-1``.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called yet.
        """
        self._check_fitted()
        df = df.copy()

        for col, mapping in self._mappings.items():
            if col not in df.columns:
                logger.warning(
                    "LabelEncoderStep transform: column '%s' not found — skipping.", col
                )
                continue

            df[col] = df[col].map(mapping).fillna(-1).astype(int)
            logger.debug("LabelEncoderStep: label-encoded column '%s'.", col)

        return df
