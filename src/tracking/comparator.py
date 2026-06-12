"""
comparator.py — RunComparator queries, ranks, and compares runs from ExperimentTracker.
"""

import pandas as pd
from tabulate import tabulate
from src.tracking.tracker import ExperimentTracker
from src.tracking.run_record import RunRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RunComparator:
    """Compares and ranks experiment runs from the database.

    Parameters
    ----------
    tracker : ExperimentTracker
        An instance of ExperimentTracker to fetch runs from.
    """

    def __init__(self, tracker: ExperimentTracker) -> None:
        self.tracker = tracker

    def _get_metric_value(self, run: RunRecord, metric: str) -> float | None:
        """Helper to extract a metric value from a RunRecord, resolving aliases."""
        if not run.metrics:
            return None
        
        # Standard metric aliases mapping
        aliases = {
            "f1": "f1_weighted",
            "precision": "precision_weighted",
            "recall": "recall_weighted"
        }
        
        target = aliases.get(metric.lower(), metric.lower())
        
        # Try exact matching in lower/upper or raw keys
        for key, val in run.metrics.items():
            if key.lower() == target:
                return float(val)
        return None

    def get_best(self, metric: str, top_n: int = 5) -> list[RunRecord]:
        """Rank runs by a given metric and return the top N.

        Supports: accuracy, f1, precision, recall, rmse, mae, r2.
        For rmse/mae, sorts ascending (lower is better). For others, sorts descending.

        Parameters
        ----------
        metric : str
            The metric to sort by (e.g. "f1", "rmse").
        top_n : int
            The number of top records to return.

        Returns
        -------
        list[RunRecord]
            Top N ranked RunRecords.
        """
        logger.debug("Ranking top %d runs by metric: %s", top_n, metric)
        runs = self.tracker.list_runs(limit=1000)
        
        valid_runs = []
        for run in runs:
            val = self._get_metric_value(run, metric)
            if val is not None:
                valid_runs.append((run, val))

        if not valid_runs:
            logger.warning("No runs found containing metric '%s'", metric)
            return []

        # Lower is better for error metrics (rmse, mae)
        is_error = metric.lower() in ["rmse", "mae"]
        valid_runs.sort(key=lambda x: x[1], reverse=not is_error)

        return [item[0] for item in valid_runs[:top_n]]

    def compare(self, run_ids: list[str]) -> pd.DataFrame:
        """Generate a side-by-side comparison DataFrame of the selected runs.

        Columns will include run_id, timestamp, model_type, followed by metrics.

        Parameters
        ----------
        run_ids : list[str]
            List of run IDs to compare.

        Returns
        -------
        pd.DataFrame
            DataFrame comparing the runs.
        """
        logger.debug("Generating side-by-side comparison for runs: %s", run_ids)
        rows = []
        for run_id in run_ids:
            try:
                run = self.tracker.get_run(run_id)
                row = {
                    "run_id": run.run_id,
                    "timestamp": run.timestamp,
                    "model_type": run.model_type,
                }
                # Flatten the metrics
                if run.metrics:
                    for k, v in run.metrics.items():
                        if k not in ["task", "confusion_matrix"]:
                            row[k] = v
                rows.append(row)
            except ValueError as e:
                logger.warning("Skipping comparison for missing run ID: %s", run_id)
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Order columns to have run_id, timestamp, model_type first, then the rest (metrics)
        cols = ["run_id", "timestamp", "model_type"]
        other_cols = [c for c in df.columns if c not in cols]
        df = df[cols + other_cols]
        return df

    def summary_table(self, top_n: int = 10) -> str:
        """Return a formatted ASCII table of the top N runs by primary metric.

        Automatically detects classification/regression to choose the primary metric
        (f1_weighted/rmse) and ranks accordingly.

        Parameters
        ----------
        top_n : int
            Number of top runs to show.

        Returns
        -------
        str
            A formatted ASCII table string.
        """
        logger.debug("Generating summary table for top %d runs.", top_n)
        runs = self.tracker.list_runs(limit=1000)
        if not runs:
            return "No runs logged yet."

        # Detect the primary metric from database runs
        primary_metric = "f1_weighted"
        for r in runs:
            if r.metrics:
                if "rmse" in r.metrics:
                    primary_metric = "rmse"
                    break
                elif "f1_weighted" in r.metrics:
                    primary_metric = "f1_weighted"
                    break

        logger.debug("Selected primary metric for ranking: %s", primary_metric)
        best_runs = self.get_best(primary_metric, top_n=top_n)
        if not best_runs:
            return "No valid runs found to rank."

        table_rows = []
        for rank, run in enumerate(best_runs, 1):
            metrics_parts = []
            if run.metrics:
                for k, v in run.metrics.items():
                    if k not in ["task", "confusion_matrix"]:
                        # Shorten name if needed, format float values
                        name = k.replace("_weighted", "")
                        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                        metrics_parts.append(f"{name}={val_str}")
            
            metrics_str = ", ".join(metrics_parts)
            table_rows.append([
                rank,
                run.run_id[:8],
                run.model_type,
                run.timestamp,
                metrics_str
            ])

        return tabulate(
            table_rows,
            headers=["Rank", "Run ID", "Model", "Timestamp", "Metrics"],
            tablefmt="grid"
        )
