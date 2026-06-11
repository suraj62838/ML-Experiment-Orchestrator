"""
registry.py — Maps configuration step-type strings to step classes.

All pipeline steps are registered here. Adding a new step requires:
  1. Importing the class
  2. Adding an entry to STEP_REGISTRY

Usage
-----
    from src.pipeline.registry import get_step

    step = get_step("standard_scale", {"cols": "all_numeric"})
    step.fit(X_train)
    X_val_t = step.transform(X_val)
"""

from __future__ import annotations
from typing import Any

from src.pipeline.base import PipelineStep
from src.pipeline.steps.encoders import OneHotEncoderStep, LabelEncoderStep
from src.pipeline.steps.scalers import StandardScalerStep, MinMaxScalerStep
from src.pipeline.steps.imputers import MeanImputerStep, MedianImputerStep, ModeImputerStep
from src.pipeline.steps.column_ops import DropColumnsStep, SelectColumnsStep, RenameColumnsStep
from src.pipeline.steps.feature_ops import LogTransformStep, PolynomialFeaturesStep

# ── Registry ─────────────────────────────────────────────────────────────────

STEP_REGISTRY: dict[str, type[PipelineStep]] = {
    "onehot_encode":        OneHotEncoderStep,
    "label_encode":         LabelEncoderStep,
    "standard_scale":       StandardScalerStep,
    "minmax_scale":         MinMaxScalerStep,
    "mean_impute":          MeanImputerStep,
    "median_impute":        MedianImputerStep,
    "mode_impute":          ModeImputerStep,
    "drop_columns":         DropColumnsStep,
    "select_columns":       SelectColumnsStep,
    "rename_columns":       RenameColumnsStep,
    "log_transform":        LogTransformStep,
    "polynomial_features":  PolynomialFeaturesStep,
}


def get_step(name: str, config: dict[str, Any]) -> PipelineStep:
    """Instantiate a pipeline step by its registry name.

    Parameters
    ----------
    name : str
        The step type string as it appears in ``experiment.yaml``
        (e.g. ``"standard_scale"``).
    config : dict
        Step-specific configuration dict (the same dict from YAML, minus
        the ``type`` key — or the full dict; the step ignores unknown keys).

    Returns
    -------
    PipelineStep
        An unfit instance of the corresponding step class.

    Raises
    ------
    ValueError
        If *name* is not found in :data:`STEP_REGISTRY`, with a message
        listing all available step names.

    Examples
    --------
    >>> step = get_step("mean_impute", {"cols": "all_numeric"})
    >>> type(step)
    <class 'src.pipeline.steps.imputers.MeanImputerStep'>
    """
    if name not in STEP_REGISTRY:
        available = sorted(STEP_REGISTRY.keys())
        raise ValueError(
            f"Unknown pipeline step type: '{name}'.\n"
            f"Available steps: {available}"
        )
    return STEP_REGISTRY[name](config)
