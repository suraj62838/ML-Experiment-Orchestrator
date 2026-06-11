"""
column_ops.py — Column manipulation pipeline steps.

Steps
-----
DropColumnsStep
    Drops named columns from a DataFrame. Warns if any are missing.
SelectColumnsStep
    Returns only the specified columns (stateless — no fit required).
RenameColumnsStep
    Renames columns according to a mapping dict (stateless).
"""

from __future__ import annotations
import warnings
from typing import Any
import pandas as pd
from src.pipeline.base import PipelineStep
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DropColumnsStep(PipelineStep):
    """Drop a list of named columns from the DataFrame.

    During ``fit`` the step validates which columns exist and warns about
    any that are missing.  During ``transform`` the listed columns are
    dropped; missing columns are again warned about but do not raise errors.

    Parameters
    ----------
    config : dict
        ``cols`` : list[str]  — Column names to drop.

    Examples
    --------
    >>> step = DropColumnsStep({"cols": ["id", "row_hash"]})
    >>> X = step.fit_transform(X_train)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._cols: list[str] = list(config.get("cols", []))

    def fit(self, df: pd.DataFrame) -> "DropColumnsStep":
        """Validate that configured columns exist in the training DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Training data (used only for column validation).

        Returns
        -------
        DropColumnsStep
            Returns ``self`` for chaining.
        """
        missing = [c for c in self._cols if c not in df.columns]
        if missing:
            warnings.warn(
                f"DropColumnsStep: the following columns to drop were not found "
                f"in the DataFrame: {missing}. They will be silently skipped.",
                UserWarning,
                stacklevel=2,
            )
        self._fitted = True
        logger.info("DropColumnsStep fitted — will drop: %s", self._cols)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop the configured columns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with the target columns removed.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called.
        """
        self._check_fitted()
        df = df.copy()
        present = [c for c in self._cols if c in df.columns]
        skipped = set(self._cols) - set(present)
        if skipped:
            logger.warning("DropColumnsStep: skipping missing columns: %s", sorted(skipped))
        df = df.drop(columns=present)
        logger.info("DropColumnsStep: dropped %d column(s): %s", len(present), present)
        return df


class SelectColumnsStep(PipelineStep):
    """Return only the specified columns from the DataFrame.

    This step is effectively stateless — ``fit`` is a no-op that simply
    marks the step as fitted.

    Parameters
    ----------
    config : dict
        ``cols`` : list[str]  — Column names to retain.

    Examples
    --------
    >>> step = SelectColumnsStep({"cols": ["age", "income", "score"]})
    >>> X_selected = step.fit_transform(X_train)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._cols: list[str] = list(config.get("cols", []))

    def fit(self, df: pd.DataFrame) -> "SelectColumnsStep":
        """Mark step as fitted (no parameters to learn).

        Parameters
        ----------
        df : pd.DataFrame
            Training data (used only to warn about missing columns).

        Returns
        -------
        SelectColumnsStep
            Returns ``self`` for chaining.
        """
        missing = [c for c in self._cols if c not in df.columns]
        if missing:
            warnings.warn(
                f"SelectColumnsStep: the following columns were not found: {missing}.",
                UserWarning,
                stacklevel=2,
            )
        self._fitted = True
        logger.info("SelectColumnsStep fitted — will keep: %s", self._cols)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only the selected columns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Copy of *df* containing only the configured columns.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called.
        """
        self._check_fitted()
        present = [c for c in self._cols if c in df.columns]
        missing = set(self._cols) - set(present)
        if missing:
            logger.warning("SelectColumnsStep: columns not found, skipping: %s", sorted(missing))
        result = df[present].copy()
        logger.info("SelectColumnsStep: selected %d column(s).", len(present))
        return result


class RenameColumnsStep(PipelineStep):
    """Rename columns according to a ``{old_name: new_name}`` mapping.

    This step is stateless — ``fit`` is a no-op.  Columns not present in
    the DataFrame are silently skipped (``pandas`` rename behaviour).

    Parameters
    ----------
    config : dict
        ``mapping`` : dict[str, str]  — Mapping of old → new column names.

    Examples
    --------
    >>> step = RenameColumnsStep({"mapping": {"fname": "first_name"}})
    >>> X_renamed = step.fit_transform(X_train)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._mapping: dict[str, str] = dict(config.get("mapping", {}))

    def fit(self, df: pd.DataFrame) -> "RenameColumnsStep":
        """Mark step as fitted (no parameters to learn).

        Parameters
        ----------
        df : pd.DataFrame
            Training data (used only to validate mapping keys).

        Returns
        -------
        RenameColumnsStep
            Returns ``self`` for chaining.
        """
        missing = [k for k in self._mapping if k not in df.columns]
        if missing:
            warnings.warn(
                f"RenameColumnsStep: these source columns were not found: {missing}.",
                UserWarning,
                stacklevel=2,
            )
        self._fitted = True
        logger.info("RenameColumnsStep fitted — mapping: %s", self._mapping)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the column rename mapping.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with renamed columns.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called.
        """
        self._check_fitted()
        df = df.copy()
        # Only rename columns that actually exist
        active_map = {k: v for k, v in self._mapping.items() if k in df.columns}
        df = df.rename(columns=active_map)
        logger.info("RenameColumnsStep: applied %d rename(s): %s", len(active_map), active_map)
        return df
