"""
base.py — Abstract base class for all pipeline transformation steps.

Every step in the ML pipeline inherits from ``PipelineStep``.  The contract
enforces that ``fit`` and ``transform`` are always separate operations, which
is critical for preventing data leakage: fitting parameters are learned from
training data only, then applied independently to validation and test sets.

Usage
-----
    class MyStep(PipelineStep):
        def fit(self, df: pd.DataFrame) -> "MyStep":
            # learn parameters from df (train only)
            self._fitted = True
            return self

        def transform(self, df: pd.DataFrame) -> pd.DataFrame:
            self._check_fitted()
            # apply learned parameters to df
            return df.copy()
"""

from __future__ import annotations

import abc
from typing import Any

import pandas as pd


class NotFittedError(RuntimeError):
    """Raised when :py:meth:`transform` is called before :py:meth:`fit`."""


class PipelineStep(abc.ABC):
    """Abstract base class for all pipeline transformation steps.

    Subclasses must implement :py:meth:`fit` and :py:meth:`transform`.
    Every step receives its configuration through the constructor ``config``
    dict and stores all fitted parameters as **instance** attributes (never
    class-level), so multiple engine instances never share state.

    Parameters
    ----------
    config : dict
        Step-specific configuration parsed from ``experiment.yaml``.
        Keys and semantics are defined by each concrete subclass.

    Notes
    -----
    * ``fit`` must only be called on **training** data.
    * ``transform`` must use only the parameters learned in ``fit``; it
      must **never** look at the full dataset or recalculate statistics.
    * Every ``transform`` implementation must work on a **copy** of the
      input DataFrame and must not modify the original.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Store the configuration dict and initialise the fitted flag."""
        self.config: dict[str, Any] = config
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def fit(self, df: pd.DataFrame) -> "PipelineStep":
        """Learn parameters from *df* (training data only).

        Parameters
        ----------
        df : pd.DataFrame
            Training-split feature DataFrame (target column already removed).

        Returns
        -------
        PipelineStep
            Returns ``self`` to allow chaining: ``step.fit(df).transform(df)``.

        Notes
        -----
        Implementations must set ``self._fitted = True`` before returning.
        """

    @abc.abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted transformation to *df* and return a new DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to transform.  May be train, val, or test split.

        Returns
        -------
        pd.DataFrame
            A transformed **copy** of *df*.  The original must not be mutated.

        Raises
        ------
        NotFittedError
            If called before :py:meth:`fit`.
        """

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on *df* then immediately transform and return the result.

        This is a convenience wrapper — equivalent to calling
        ``step.fit(df).transform(df)``.  Only use this on training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training-split feature DataFrame.

        Returns
        -------
        pd.DataFrame
            Transformed copy of *df*.
        """
        return self.fit(df).transform(df)

    def _check_fitted(self) -> None:
        """Raise :class:`NotFittedError` if this step has not been fitted yet.

        Call this at the start of every :py:meth:`transform` implementation.
        """
        if not self._fitted:
            raise NotFittedError(
                f"{self.__class__.__name__} has not been fitted yet. "
                "Call fit() on training data before calling transform()."
            )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return f"{self.__class__.__name__}(config={self.config!r}, status={status})"
