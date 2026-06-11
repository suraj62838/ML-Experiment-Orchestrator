"""
engine.py — PipelineEngine: reads a step config list and chains steps.

The engine is the central orchestrator of the transformation pipeline.
It reads a list of step configs from experiment.yaml, instantiates each
step via the registry, fits them on training data only, and applies them
in order to any dataset.

Usage
-----
    from src.pipeline.engine import PipelineEngine

    engine = PipelineEngine(config["pipeline"]["steps"])
    X_train_t = engine.fit_transform(X_train)
    X_val_t   = engine.transform(X_val)
    X_test_t  = engine.transform(X_test)

CRITICAL
--------
    engine.fit() / engine.fit_transform() must only be called on training
    data. Calling transform() on val/test data re-uses the parameters
    learned from training, preventing data leakage.
"""

from __future__ import annotations
from typing import Any

import pandas as pd

from src.pipeline.base import PipelineStep
from src.pipeline.registry import get_step
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PipelineEngine:
    """Orchestrates a sequence of :class:`~src.pipeline.base.PipelineStep` objects.

    The engine reads a list of step configurations (parsed from the
    ``pipeline.steps`` section of ``experiment.yaml``) and manages the full
    fit/transform lifecycle.

    Parameters
    ----------
    steps_config : list[dict]
        A list of step configuration dicts.  Each dict must contain at
        least a ``"type"`` key matching a name in :data:`~src.pipeline.registry.STEP_REGISTRY`.
        All other keys are forwarded to the step constructor as its config.

    Example YAML section
    --------------------
    .. code-block:: yaml

        pipeline:
          steps:
            - type: drop_columns
              cols: ["id"]
            - type: mean_impute
              cols: "all_numeric"
            - type: standard_scale
              cols: "all_numeric"

    Examples
    --------
    >>> engine = PipelineEngine(config["pipeline"]["steps"])
    >>> X_train_t = engine.fit_transform(X_train)
    >>> X_test_t  = engine.transform(X_test)
    """

    def __init__(self, steps_config: list[dict[str, Any]]) -> None:
        """Store the step configurations; steps are not instantiated yet."""
        self._steps_config: list[dict[str, Any]] = steps_config
        self._steps: list[PipelineStep] = []
        self._step_names: list[str] = []
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train: pd.DataFrame) -> "PipelineEngine":
        """Instantiate each step and fit it on training data in sequence.

        Steps are created fresh from the config on every ``fit`` call,
        so the engine can be re-fit safely.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature DataFrame (target column must already be removed).

        Returns
        -------
        PipelineEngine
            Returns ``self`` for chaining.

        Raises
        ------
        ValueError
            If any step type string is not found in the registry.
        """
        self._steps = []
        self._step_names = []
        df = X_train.copy()

        logger.info("PipelineEngine fitting %d step(s)…", len(self._steps_config))

        for step_cfg in self._steps_config:
            step_cfg = dict(step_cfg)           # shallow copy — don't mutate config
            step_type = step_cfg.get("type", "")
            step = get_step(step_type, step_cfg)

            in_shape = df.shape
            step.fit(df)
            df = step.transform(df)
            out_shape = df.shape

            self._steps.append(step)
            self._step_names.append(step_type)

            logger.info(
                "Fit step %-20s | %s -> %s", step_type, in_shape, out_shape
            )

        self._fitted = True
        logger.info("PipelineEngine fit complete.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply each fitted step's transform in order.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame to transform (may be val or test split).

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame.

        Raises
        ------
        RuntimeError
            If ``fit`` has not been called yet.
        """
        if not self._fitted:
            raise RuntimeError(
                "PipelineEngine has not been fitted. "
                "Call fit() or fit_transform() on training data first."
            )

        df = df.copy()
        for name, step in zip(self._step_names, self._steps):
            in_shape = df.shape
            df = step.transform(df)
            out_shape = df.shape
            logger.debug("Transform step %-20s | %s -> %s", name, in_shape, out_shape)

        return df

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Fit the engine on training data and return the transformed result.

        Equivalent to calling ``fit(X_train)`` then ``transform(X_train)``,
        but more efficient because fit already computes the transformation.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature DataFrame.

        Returns
        -------
        pd.DataFrame
            Transformed training DataFrame.
        """
        self.fit(X_train)
        # Re-transform from scratch so the transform path is exercised
        return self.transform(X_train)

    def get_steps_summary(self) -> list[dict[str, Any]]:
        """Return a structured summary of all configured steps.

        Returns
        -------
        list[dict]
            Each entry is ``{"name": str, "config": dict}`` for every
            step in the pipeline (in order).  Useful for logging and
            experiment reports.
        """
        return [
            {"name": name, "config": cfg}
            for name, cfg in zip(self._step_names, [s.config for s in self._steps])
        ]

    def print_summary(
        self,
        X_train: pd.DataFrame,
        X_train_t: pd.DataFrame,
    ) -> None:
        """Print the pipeline step-by-step shape table to stdout.

        Runs a shape-tracking pass over the training data and prints a
        formatted summary table.

        Parameters
        ----------
        X_train : pd.DataFrame
            Original (pre-transform) training DataFrame.
        X_train_t : pd.DataFrame
            Transformed training DataFrame (output of fit_transform).
        """
        if not self._fitted:
            return

        # Re-run transform step-by-step to capture intermediate shapes
        shapes: list[tuple[tuple, tuple]] = []
        df_curr = X_train.copy()
        for step in self._steps:
            in_shape = df_curr.shape
            df_curr = step.transform(df_curr)
            out_shape = df_curr.shape
            shapes.append((in_shape, out_shape))

        header = "===== PIPELINE ====="
        footer = "=" * len(header)
        print(f"\n{header}")
        for i, (name, (in_s, out_s)) in enumerate(
            zip(self._step_names, shapes), start=1
        ):
            print(f"  Step {i:<2} | {name:<22} | {str(in_s):<14} -> {out_s}")
        print(f"{footer}\n")
