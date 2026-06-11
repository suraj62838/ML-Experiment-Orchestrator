"""
run.py — Main entry point for the ML Experiment Orchestrator.

Usage
-----
    python run.py --config configs/experiment.yaml

The script orchestrates the full ML pipeline:
    1. Load configuration from YAML
    2. Generate a dummy dataset (if no real data file exists) for demonstration
    3. Load data
    4. Clean data
    5. Split data into train / val / test sets
    6. Train the model
    7. Evaluate the model
    8. Save the evaluation report
    9. Print a formatted metrics summary table

All steps are logged to both the console and ``run.log``.
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
from src.data.cleaner import DataCleaner
from src.data.splitter import DataSplitter
from src.models.trainer import ModelTrainer
from src.models.evaluator import Evaluator

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _generate_dummy_dataset(filepath: str, n_samples: int = 300, random_seed: int = 42) -> None:
    """Create a small synthetic CSV dataset for demonstration purposes.

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

    df = pd.DataFrame(
        {
            "feature_1": rng.standard_normal(n_samples),
            "feature_2": rng.standard_normal(n_samples),
            "feature_3": rng.uniform(0, 10, size=n_samples),
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
    cleaning_cfg = config.get("cleaning", {})
    splitting_cfg = config.get("splitting", {})
    model_cfg = config["model"]
    eval_cfg = config["evaluation"]

    filepath: str = data_cfg["filepath"]
    target_col: str = data_cfg["target_col"]

    # ── 2. Generate dummy data if none exists ───────────────────────────────
    if not Path(filepath).exists():
        logger.warning(
            "Data file '%s' not found. Generating a synthetic dummy dataset…",
            filepath,
        )
        seed = splitting_cfg.get("random_seed", 42)
        _generate_dummy_dataset(filepath, n_samples=300, random_seed=seed)

    # ── 3. Load data ────────────────────────────────────────────────────────
    logger.info("-- Step 1/6: Loading data --")
    loader = DataLoader()
    df = loader.load(filepath)

    # ── 4. Clean data ───────────────────────────────────────────────────────
    logger.info("-- Step 2/6: Cleaning data --")
    cleaner = DataCleaner(cleaning_cfg)
    df_clean = cleaner.clean(df)

    # ── 5. Split data ───────────────────────────────────────────────────────
    logger.info("-- Step 3/6: Splitting data --")
    splitter = DataSplitter(
        test_size=splitting_cfg.get("test_size", 0.2),
        val_size=splitting_cfg.get("val_size", 0.1),
        random_seed=splitting_cfg.get("random_seed", 42),
        stratify=splitting_cfg.get("stratify", False),
    )
    splits = splitter.split(df_clean, target_col)

    # ── 6. Train model ──────────────────────────────────────────────────────
    logger.info("-- Step 4/6: Training model --")
    trainer = ModelTrainer(
        model_type=model_cfg["type"],
        model_params=model_cfg.get("params", {}),
    )
    trainer.fit(splits["X_train"], splits["y_train"])
    trainer.save(model_cfg["save_path"])

    # ── 7. Evaluate model ───────────────────────────────────────────────────
    logger.info("-- Step 5/6: Evaluating model --")
    evaluator = Evaluator()
    metrics = evaluator.evaluate(trainer.model, splits["X_test"], splits["y_test"])

    # ── 8. Save report ──────────────────────────────────────────────────────
    logger.info("-- Step 6/6: Saving report --")
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
