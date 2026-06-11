"""
trainer.py — Wraps scikit-learn estimators with a unified fit/save/load API.

Supported model types (extend ``_MODEL_REGISTRY`` to add more):
  - "logistic_regression"
  - "random_forest"
  - "decision_tree"
  - "ridge"
  - "lasso"
  - "svr"
  - "svc"
"""

import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Model registry ────────────────────────────────────────────────────────────
# Maps configuration string → sklearn estimator class.
# Add new model types here without touching the rest of the code.
_MODEL_REGISTRY: dict[str, type] = {
    "logistic_regression": LogisticRegression,
    "random_forest_classifier": RandomForestClassifier,
    "random_forest_regressor": RandomForestRegressor,
    "random_forest": RandomForestClassifier,   # convenience alias
    "decision_tree": DecisionTreeClassifier,
    "ridge": Ridge,
    "lasso": Lasso,
    "svc": SVC,
    "svr": SVR,
}


class ModelTrainer:
    """Instantiates, trains, and serialises an sklearn estimator.

    Parameters
    ----------
    model_type : str
        Key from the ``_MODEL_REGISTRY`` dict that selects which sklearn
        estimator to use (e.g. ``"logistic_regression"``).
    model_params : dict, optional
        Keyword arguments forwarded to the estimator's constructor
        (e.g. ``{"C": 0.5, "max_iter": 500}``).

    Attributes
    ----------
    model : sklearn estimator | None
        The fitted model. ``None`` until :meth:`fit` is called.

    Example
    -------
    >>> trainer = ModelTrainer("logistic_regression", {"max_iter": 1000})
    >>> trainer.fit(X_train, y_train)
    >>> trainer.save("outputs/model.joblib")
    """

    def __init__(self, model_type: str, model_params: dict | None = None) -> None:
        """Resolve *model_type* to an sklearn class and instantiate it."""
        if model_type not in _MODEL_REGISTRY:
            supported = sorted(_MODEL_REGISTRY.keys())
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Supported types: {supported}."
            )

        self._model_type = model_type
        estimator_cls = _MODEL_REGISTRY[model_type]
        params = model_params or {}
        self.model = estimator_cls(**params)
        logger.info("Initialised model: %s (params=%s)", estimator_cls.__name__, params)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train, y_train) -> "ModelTrainer":
        """Fit the model on the given training data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y_train : array-like of shape (n_samples,)
            Training target vector.

        Returns
        -------
        ModelTrainer
            Returns *self* so calls can be chained.
        """
        logger.info("Training %s on %d samples…", type(self.model).__name__, len(y_train))
        self.model.fit(X_train, y_train)
        logger.info("Training complete.")
        return self

    def save(self, path: str) -> None:
        """Serialise the fitted model to disk using :mod:`joblib`.

        Parameters
        ----------
        path : str
            Destination file path (e.g. ``"outputs/model.joblib"``).
            Parent directories are created automatically.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called before :meth:`save`.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before save().")

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, out_path)
        logger.info("Model saved to '%s'.", out_path)

    def load(self, path: str) -> "ModelTrainer":
        """Deserialise a model from disk and store it as ``self.model``.

        Parameters
        ----------
        path : str
            Path to a joblib-serialised model file.

        Returns
        -------
        ModelTrainer
            Returns *self* so the trainer can be used immediately after loading.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: '{path}'.")

        self.model = joblib.load(load_path)
        logger.info("Model loaded from '%s'.", load_path)
        return self
