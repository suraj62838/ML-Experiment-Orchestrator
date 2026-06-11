"""
cleaner.py — Applies configurable cleaning steps to a DataFrame.

Supported cleaning strategies (set via constructor config dict):
  fill_nulls      : "mean" | "median" | "mode" | "drop"
  drop_duplicates : true | false
  fix_dtypes      : true | false  (coerce object columns to numeric where possible)
"""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Valid fill-null strategies
_FILL_STRATEGIES = frozenset({"mean", "median", "mode", "drop"})


class DataCleaner:
    """Cleans a DataFrame according to a user-supplied configuration.

    Parameters
    ----------
    config : dict
        A dict with the following optional keys:

        ``fill_nulls`` : str, default ``"mean"``
            Strategy for handling missing values.
            One of ``"mean"``, ``"median"``, ``"mode"``, or ``"drop"``.
        ``drop_duplicates`` : bool, default ``True``
            Whether to remove fully duplicate rows.
        ``fix_dtypes`` : bool, default ``True``
            Whether to attempt numeric coercion on object columns.

    Example
    -------
    >>> cfg = {"fill_nulls": "median", "drop_duplicates": True, "fix_dtypes": True}
    >>> cleaner = DataCleaner(cfg)
    >>> df_clean = cleaner.clean(df)
    """

    def __init__(self, config: dict) -> None:
        """Initialise the cleaner with the given configuration dict."""
        self._fill_nulls: str = config.get("fill_nulls", "mean")
        self._drop_duplicates: bool = bool(config.get("drop_duplicates", True))
        self._fix_dtypes: bool = bool(config.get("fix_dtypes", True))

        if self._fill_nulls not in _FILL_STRATEGIES:
            raise ValueError(
                f"Unknown fill_nulls strategy '{self._fill_nulls}'. "
                f"Valid options: {sorted(_FILL_STRATEGIES)}."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all configured cleaning steps and return a new DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The raw input DataFrame (not modified in place).

        Returns
        -------
        pd.DataFrame
            A cleaned copy of *df*.
        """
        df = df.copy()

        if self._drop_duplicates:
            df = self._apply_drop_duplicates(df)

        if self._fix_dtypes:
            df = self._apply_fix_dtypes(df)

        df = self._apply_fill_nulls(df)

        logger.info("Cleaning complete. Final shape: %s", df.shape)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove fully duplicate rows and log the count removed."""
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        logger.info("drop_duplicates: removed %d duplicate row(s).", removed)
        return df

    def _apply_fix_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce object-typed columns to numeric where possible."""
        coerced: list[str] = []
        for col in df.select_dtypes(include="object").columns:
            converted = pd.to_numeric(df[col], errors="coerce")
            # Only apply if at least one value was successfully converted
            if converted.notna().sum() > 0:
                df[col] = converted
                coerced.append(col)
        if coerced:
            logger.info("fix_dtypes: coerced columns to numeric: %s.", coerced)
        else:
            logger.info("fix_dtypes: no object columns needed coercion.")
        return df

    def _apply_fill_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill (or drop) null values according to the configured strategy."""
        null_counts = df.isnull().sum()
        total_nulls = int(null_counts.sum())

        if total_nulls == 0:
            logger.info("fill_nulls: no null values found — nothing to fill.")
            return df

        strategy = self._fill_nulls
        numeric_cols = df.select_dtypes(include="number").columns

        if strategy == "drop":
            before = len(df)
            df = df.dropna()
            logger.info(
                "fill_nulls (drop): dropped %d row(s) containing nulls.",
                before - len(df),
            )
        elif strategy == "mean":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            logger.info(
                "fill_nulls (mean): filled %d null(s) in numeric columns.", total_nulls
            )
        elif strategy == "median":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            logger.info(
                "fill_nulls (median): filled %d null(s) in numeric columns.",
                total_nulls,
            )
        elif strategy == "mode":
            for col in df.columns:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val.iloc[0])
            logger.info(
                "fill_nulls (mode): filled %d null(s) across all columns.", total_nulls
            )

        return df
