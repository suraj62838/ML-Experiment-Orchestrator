"""
evaluator.py — Computes and persists performance metrics for any sklearn model.

Task type (classification vs. regression) is detected automatically from the
model's class by inspecting whether it exposes a ``predict_proba`` method and
whether it descends from sklearn's ``ClassifierMixin``.

Classification metrics : accuracy, f1 (weighted), precision (weighted),
                          recall (weighted), confusion_matrix
Regression metrics     : MAE, RMSE, R²
"""

import json
from pathlib import Path

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Evaluates a fitted sklearn estimator and persists the resulting metrics.

    Task type is inferred from the estimator class — no manual flag needed.

    Example
    -------
    >>> evaluator = Evaluator()
    >>> metrics = evaluator.evaluate(trainer.model, X_test, y_test)
    >>> evaluator.save_report(metrics, "outputs/report.json")
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, model, X_test, y_test) -> dict:
        """Compute performance metrics appropriate for the model's task type.

        Parameters
        ----------
        model : fitted sklearn estimator
            Any fitted estimator that exposes a ``predict`` method.
        X_test : array-like of shape (n_samples, n_features)
            Test feature matrix.
        y_test : array-like of shape (n_samples,)
            True target values for the test set.

        Returns
        -------
        dict
            A flat dictionary of metric names → scalar values (or lists for
            the confusion matrix).

        Raises
        ------
        TypeError
            If *model* is neither a ``ClassifierMixin`` nor a
            ``RegressorMixin``.
        """
        y_pred = model.predict(X_test)

        if isinstance(model, ClassifierMixin):
            metrics = self._classification_metrics(y_test, y_pred)
            logger.info("Evaluation (classification): %s", metrics)
        elif isinstance(model, RegressorMixin):
            metrics = self._regression_metrics(y_test, y_pred)
            logger.info("Evaluation (regression): %s", metrics)
        else:
            raise TypeError(
                f"Cannot determine task type for model of class "
                f"'{type(model).__name__}'. Expected a ClassifierMixin or "
                f"RegressorMixin."
            )

        return metrics

    def save_report(self, metrics: dict, path: str) -> None:
        """Serialise *metrics* to a JSON file at *path*.

        Parameters
        ----------
        metrics : dict
            The dict returned by :meth:`evaluate`.
        path : str
            Destination file path (e.g. ``"outputs/report.json"``).
            Parent directories are created automatically.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)

        logger.info("Metrics report saved to '%s'.", out_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classification_metrics(y_true, y_pred) -> dict:
        """Compute classification metrics and return them as a dict."""
        cm = confusion_matrix(y_true, y_pred)
        return {
            "task": "classification",
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
            "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
            "precision_weighted": round(
                float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 6
            ),
            "recall_weighted": round(
                float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 6
            ),
            "confusion_matrix": cm.tolist(),
        }

    @staticmethod
    def _regression_metrics(y_true, y_pred) -> dict:
        """Compute regression metrics and return them as a dict."""
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return {
            "task": "regression",
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 6),
            "rmse": round(rmse, 6),
            "r2": round(float(r2_score(y_true, y_pred)), 6),
        }
