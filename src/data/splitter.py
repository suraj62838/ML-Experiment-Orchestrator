"""
splitter.py — Splits a DataFrame into train / validation / test sets.

The splitting is always seeded for reproducibility.
Stratified splitting is supported for classification tasks.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataSplitter:
    """Splits a labelled DataFrame into train, validation, and test partitions.

    Parameters
    ----------
    test_size : float
        Fraction of the full dataset reserved for the test set (e.g. ``0.2``).
    val_size : float
        Fraction of the *training remainder* reserved for the validation set.
        E.g., with ``test_size=0.2`` and ``val_size=0.1`` the final split is
        roughly 72 % train / 8 % val / 20 % test.
    random_seed : int
        Random state for all ``train_test_split`` calls to ensure
        reproducible results.
    stratify : bool
        Whether to perform stratified splitting based on the target column.
        Set to ``True`` for classification tasks.

    Example
    -------
    >>> splitter = DataSplitter(test_size=0.2, val_size=0.1,
    ...                         random_seed=42, stratify=True)
    >>> sets = splitter.split(df, target_col="label")
    >>> X_train, y_train = sets["X_train"], sets["y_train"]
    """

    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_seed: int = 42,
        stratify: bool = False,
    ) -> None:
        """Initialise the splitter with partition sizes and a random seed."""
        self._test_size = test_size
        self._val_size = val_size
        self._random_seed = random_seed
        self._stratify = stratify

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(self, df: pd.DataFrame, target_col: str) -> dict[str, pd.DataFrame | pd.Series]:
        """Split *df* into six partitions: X/y for train, val, and test.

        Parameters
        ----------
        df : pd.DataFrame
            The full cleaned dataset (features + target column).
        target_col : str
            Name of the column that holds the prediction target.

        Returns
        -------
        dict
            Keys: ``X_train``, ``X_val``, ``X_test``,
                  ``y_train``, ``y_val``, ``y_test``.

        Raises
        ------
        KeyError
            If *target_col* is not found in *df*.
        """
        if target_col not in df.columns:
            raise KeyError(
                f"Target column '{target_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}."
            )

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # ── Step 1: carve out the test set ──────────────────────────────
        stratify_labels = y if self._stratify else None
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=self._test_size,
            random_state=self._random_seed,
            stratify=stratify_labels,
        )

        # ── Step 2: split the remainder into train / val ─────────────────
        # val_size is expressed relative to the *original* dataset, so we
        # must adjust it relative to the temp (non-test) slice.
        relative_val_size = self._val_size / (1.0 - self._test_size)
        stratify_temp = y_temp if self._stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=relative_val_size,
            random_state=self._random_seed,
            stratify=stratify_temp,
        )

        logger.info(
            "Split complete — train: %d | val: %d | test: %d",
            len(X_train),
            len(X_val),
            len(X_test),
        )

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }
