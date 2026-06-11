"""
run.py — Main entry point for the ML Experiment Orchestrator (Phase 2).

Usage
-----
    python run.py --config configs/experiment.yaml

The script orchestrates the full ML pipeline:
    1. Load configuration from YAML
    2. Generate a dummy dataset (if no real data file exists) for demonstration
    3. Load data
    4. Split data into train / val / test sets  ← BEFORE pipeline (prevents leakage)
    5. Build PipelineEngine from config pipeline.steps
    6. engine.fit_transform(X_train)  → X_train_transformed
    7. engine.transform(X_val)        → X_val_transformed
    8. engine.transform(X_test)       → X_test_transformed
    9. Train model on X_train_transformed
    10. Evaluate on X_test_transformed
    11. Save report and print pipeline summary + metrics table

CRITICAL: The pipeline is fit on X_train ONLY.  Fitting on val/test data
would constitute data leakage and invalidate the evaluation.
"""

import argparse
import sys
import os
from pathlib import Path

# ── Ensure project root is on the Python path so `src` is importable ─────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.data.loader import DataLoader
from src.data.splitter import DataSplitter
from src.models.trainer import ModelTrainer
from src.models.evaluator import Evaluator
from src.pipeline.engine import PipelineEngine

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _generate_dummy_dataset(
    filepath: str, n_samples: int = 300, random_seed: int = 42
) -> None:
    """Create a synthetic CSV dataset with numeric and categorical columns.

    The generated dataset is intentionally rich so that every configured
    pipeline step (drop_columns, imputers, onehot_encode, standard_scale)
    has real data to act on.

    Parameters
    ----------
    filepath : str
        Destination path for the generated CSV (parent dirs are created).
    n_samples : int
        Number of rows to generate.
    random_seed : int
        NumPy random seed for reproducibility.
    """
    rng = np.random.default_rng(random_seed)
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    genders = rng.choice(["male", "female", "other"], size=n_samples)
    categories = rng.choice(["A", "B", "C"], size=n_samples)

    # Introduce ~10 % nulls in numeric columns to exercise imputers
    feature_1 = rng.standard_normal(n_samples)
    feature_2 = rng.standard_normal(n_samples)
    feature_3 = rng.uniform(0, 10, size=n_samples)

    null_mask_1 = rng.random(n_samples) < 0.1
    null_mask_2 = rng.random(n_samples) < 0.1
    feature_1[null_mask_1] = np.nan
    feature_2[null_mask_2] = np.nan

    # Introduce ~10 % nulls in categorical columns for mode imputer
    gender_null = rng.random(n_samples) < 0.1
    cat_null = rng.random(n_samples) < 0.1
    genders_series = pd.array(genders, dtype=object)
    categories_series = pd.array(categories, dtype=object)
    genders_series[gender_null] = None
    categories_series[cat_null] = None

    df = pd.DataFrame(
        {
            "id": range(1, n_samples + 1),       # surrogate key — will be dropped
            "gender": genders_series,
            "category": categories_series,
            "feature_1": feature_1,
            "feature_2": feature_2,
            "feature_3": feature_3,
            "feature_4": rng.integers(0, 5, size=n_samples).astype(float),
            "target": rng.integers(0, 2, size=n_samples),
        }
    )
    df.to_csv(out_path, index=False)
    logger.info("Generated dummy dataset at '%s' (%d rows).", out_path, n_samples)


def _print_metrics_table(metrics: dict) -> None:
    """Pretty-print a metrics dict as an ASCII summary table.

    Parameters
    ----------
    metrics : dict
        The dict returned by :class:`~src.models.evaluator.Evaluator`.
    """
    border = "=" * 48
    print(f"\n{border}")
    print(f"  ML EXPERIMENT ORCHESTRATOR — RESULTS SUMMARY")
    print(border)
    for key, value in metrics.items():
        if key == "confusion_matrix":
            print(f"  {'confusion_matrix':<26} {value}")
        elif key == "task":
            print(f"  {'task':<26} {value}")
        else:
            print(f"  {key:<26} {value:.6f}")
    print(f"{border}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(config: dict) -> dict:
    """Execute the full ML pipeline defined by *config*.

    Parameters
    ----------
    config : dict
        Parsed experiment configuration (see ``configs/experiment.yaml``).

    Returns
    -------
    dict
        The evaluation metrics produced by :class:`~src.models.evaluator.Evaluator`.
    """
    # ── 1. Resolve config sections ──────────────────────────────────────────
    data_cfg = config["data"]
    pipeline_cfg = config.get("pipeline", {})
    splitting_cfg = config.get("splitting", {})
    model_cfg = config["model"]
    eval_cfg = config["evaluation"]

    filepath: str = data_cfg["filepath"]
    target_col: str = data_cfg["target_col"]
    steps_config: list = pipeline_cfg.get("steps", [])

    # ── 2. Generate dummy data if none exists ───────────────────────────────
    if not Path(filepath).exists():
        logger.warning(
            "Data file '%s' not found. Generating a synthetic dummy dataset…",
            filepath,
        )
        seed = splitting_cfg.get("random_seed", 42)
        _generate_dummy_dataset(filepath, n_samples=300, random_seed=seed)

    # ── 3. Load data ────────────────────────────────────────────────────────
    logger.info("-- Step 1/7: Loading data --")
    loader = DataLoader()
    df = loader.load(filepath)

    # ── 4. Split BEFORE pipeline (prevents data leakage) ───────────────────
    logger.info("-- Step 2/7: Splitting data (before pipeline) --")
    splitter = DataSplitter(
        test_size=splitting_cfg.get("test_size", 0.2),
        val_size=splitting_cfg.get("val_size", 0.1),
        random_seed=splitting_cfg.get("random_seed", 42),
        stratify=splitting_cfg.get("stratify", False),
    )
    splits = splitter.split(df, target_col)
    X_train, X_val, X_test = splits["X_train"], splits["X_val"], splits["X_test"]
    y_train, y_val, y_test = splits["y_train"], splits["y_val"], splits["y_test"]

    # ── 5. Build and fit PipelineEngine on X_train only ────────────────────
    logger.info("-- Step 3/7: Fitting pipeline on X_train --")
    engine = PipelineEngine(steps_config)
    X_train_t = engine.fit_transform(X_train)

    # ── 6. Transform val and test using fitted parameters ───────────────────
    logger.info("-- Step 4/7: Transforming X_val and X_test --")
    X_val_t = engine.transform(X_val)
    X_test_t = engine.transform(X_test)

    # ── 7. Print pipeline summary ───────────────────────────────────────────
    engine.print_summary(X_train, X_train_t)

    # ── 8. Train model ──────────────────────────────────────────────────────
    logger.info("-- Step 5/7: Training model --")
    trainer = ModelTrainer(
        model_type=model_cfg["type"],
        model_params=model_cfg.get("params", {}),
    )
    trainer.fit(X_train_t, y_train)
    trainer.save(model_cfg["save_path"])

    # ── 9. Evaluate model ───────────────────────────────────────────────────
    logger.info("-- Step 6/7: Evaluating model --")
    evaluator = Evaluator()
    metrics = evaluator.evaluate(trainer.model, X_test_t, y_test)

    # ── 10. Save report ─────────────────────────────────────────────────────
    logger.info("-- Step 7/7: Saving report --")
    evaluator.save_report(metrics, eval_cfg["report_path"])

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] | None
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
        Parsed arguments with a ``.config`` attribute.
    """
    parser = argparse.ArgumentParser(
        description="ML Experiment Orchestrator — run the full ML pipeline from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to the experiment YAML configuration file.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Entry point: parse args, load config, run pipeline, print summary."""
    args = _parse_args(argv)

    logger.info("Loading configuration from '%s'…", args.config)
    config = load_config(args.config)
    logger.info("Configuration loaded successfully.")

    metrics = run_pipeline(config)
    _print_metrics_table(metrics)


if __name__ == "__main__":
    main()
