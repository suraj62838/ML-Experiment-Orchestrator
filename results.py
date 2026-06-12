"""
results.py — Standalone CLI tool to query, compare, and manage experiment runs.
"""

import argparse
import sys
from pathlib import Path
from tabulate import tabulate

from src.utils.config import load_config
from src.tracking import ExperimentTracker, RunComparator, ModelRegistry


def get_db_path() -> str:
    """Attempt to resolve the SQLite database path from the configuration file."""
    config_path = "configs/experiment.yaml"
    default_db = "outputs/experiments.db"
    if Path(config_path).exists():
        try:
            config = load_config(config_path)
            return config.get("tracking", {}).get("db_path", default_db)
        except Exception:
            pass
    return default_db


def list_runs(tracker: ExperimentTracker) -> None:
    """Print the last 10 logged runs as a formatted table."""
    runs = tracker.list_runs(limit=10)
    if not runs:
        print("No runs found in the database.")
        return

    table_data = []
    for r in runs:
        # Format the metrics into a readable string
        metrics_parts = []
        if r.metrics:
            for k, v in r.metrics.items():
                if k not in ["task", "confusion_matrix"]:
                    name = k.replace("_weighted", "")
                    val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                    metrics_parts.append(f"{name}={val_str}")
        metrics_str = ", ".join(metrics_parts)

        # Indicate if it is the champion
        champ_mark = "[CHAMPION]" if r.is_champion else ""

        table_data.append([
            r.run_id[:8],
            r.model_type,
            r.timestamp[:19],  # Shorten ISO format timestamp
            metrics_str,
            champ_mark
        ])

    print("\n--- Last 10 Runs ---")
    print(tabulate(
        table_data,
        headers=["Run ID", "Model Type", "Timestamp", "Metrics", "Champion"],
        tablefmt="grid"
    ))
    print()


def show_top(tracker: ExperimentTracker, metric: str, top_n: int) -> None:
    """Rank and print top N runs by a metric."""
    comparator = RunComparator(tracker)
    best_runs = comparator.get_best(metric=metric, top_n=top_n)
    if not best_runs:
        print(f"No runs with metric '{metric}' found.")
        return

    table_data = []
    for idx, r in enumerate(best_runs, 1):
        metrics_parts = []
        if r.metrics:
            for k, v in r.metrics.items():
                if k not in ["task", "confusion_matrix"]:
                    name = k.replace("_weighted", "")
                    val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                    metrics_parts.append(f"{name}={val_str}")
        metrics_str = ", ".join(metrics_parts)

        champ_mark = "[CHAMPION]" if r.is_champion else ""

        table_data.append([
            idx,
            r.run_id[:8],
            r.model_type,
            r.timestamp[:19],
            metrics_str,
            champ_mark
        ])

    print(f"\n--- Top {top_n} Runs Ranked by '{metric}' ---")
    print(tabulate(
        table_data,
        headers=["Rank", "Run ID", "Model", "Timestamp", "Metrics", "Champion"],
        tablefmt="grid"
    ))
    print()


def compare_runs(tracker: ExperimentTracker, run_ids: list[str]) -> None:
    """Print a side-by-side comparison of specific run IDs."""
    comparator = RunComparator(tracker)
    df = comparator.compare(run_ids)
    if df.empty:
        print("No valid runs found to compare.")
        return

    print("\n--- Run Comparison ---")
    print(tabulate(
        df,
        headers="keys",
        tablefmt="grid",
        showindex=False
    ))
    print()


def promote_run(tracker: ExperimentTracker, registry: ModelRegistry, run_id: str) -> None:
    """Promote a run model to champion status."""
    try:
        registry.promote(run_id)
        print(f"Successfully promoted Run ID '{run_id}' to champion!")
    except Exception as e:
        print(f"Error promoting run: {e}", file=sys.stderr)


def show_champion(registry: ModelRegistry) -> None:
    """Print the current champion model's metadata."""
    try:
        meta = registry.get_champion_meta()
        print("\n=== CURRENT CHAMPION METADATA ===")
        print(f"Run ID    : {meta.get('run_id')}")
        print(f"Timestamp : {meta.get('timestamp')}")
        print(f"Config    : {meta.get('config')}")
        print("\nMetrics:")
        for k, v in meta.get("metrics", {}).items():
            if k == "confusion_matrix":
                print(f"  {k:<20} : {v}")
            elif isinstance(v, float):
                print(f"  {k:<20} : {v:.6f}")
            else:
                print(f"  {k:<20} : {v}")
        print("=================================\n")
    except FileNotFoundError:
        print("No champion model has been promoted yet.")
    except Exception as e:
        print(f"Error loading champion metadata: {e}", file=sys.stderr)


def delete_run(tracker: ExperimentTracker, run_id: str) -> None:
    """Delete a run record from the tracking database."""
    try:
        # Verify run exists first
        _ = tracker.get_run(run_id)
        tracker.delete_run(run_id)
        print(f"Successfully deleted Run ID '{run_id}' from the database.")
    except ValueError:
        print(f"Run ID '{run_id}' not found in the database.")
    except Exception as e:
        print(f"Error deleting run: {e}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query and manage past experiments of the ML Orchestrator."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--list",
        action="store_true",
        help="Show last 10 runs as a table."
    )
    group.add_argument(
        "--top",
        type=int,
        metavar="N",
        help="Show top N runs ranked by a metric (specified by --metric)."
    )
    group.add_argument(
        "--compare",
        nargs="+",
        metavar="RUN_ID",
        help="Compare multiple runs side-by-side."
    )
    group.add_argument(
        "--promote",
        type=str,
        metavar="RUN_ID",
        help="Promote a run to the champion model registry."
    )
    group.add_argument(
        "--champion",
        action="store_true",
        help="Show the metadata of the current champion model."
    )
    group.add_argument(
        "--delete",
        type=str,
        metavar="RUN_ID",
        help="Delete a run record from the database (does not delete model file)."
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="f1",
        help="Metric to rank runs by (used with --top, default: f1)."
    )

    args = parser.parse_args()

    db_path = get_db_path()
    tracker = ExperimentTracker(db_path=db_path)
    registry = ModelRegistry(tracker=tracker)

    if args.list:
        list_runs(tracker)
    elif args.top is not None:
        show_top(tracker, args.metric, args.top)
    elif args.compare:
        compare_runs(tracker, args.compare)
    elif args.promote:
        promote_run(tracker, registry, args.promote)
    elif args.champion:
        show_champion(registry)
    elif args.delete:
        delete_run(tracker, args.delete)


if __name__ == "__main__":
    main()
