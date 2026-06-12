"""
search_spaces.py — Parameter search spaces mapping for hyperparameter tuning.
"""

from typing import Any, Callable
import optuna

# Define search spaces using parameter-suggesting functions.
# Each entry is a mapping from parameter name -> function that takes an Optuna Trial and suggests a value.
SEARCH_SPACES: dict[str, dict[str, Callable[[optuna.Trial], Any]]] = {
    "logistic_regression": {
        "C": lambda trial: trial.suggest_float("C", 0.01, 10.0, log=True),
        "max_iter": lambda trial: trial.suggest_int("max_iter", 100, 1000),
        "solver": lambda trial: trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
    },
    "random_forest": {
        "n_estimators": lambda trial: trial.suggest_int("n_estimators", 50, 300),
        "max_depth": lambda trial: trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": lambda trial: trial.suggest_int("min_samples_split", 2, 10),
    },
    "xgboost": {
        "n_estimators": lambda trial: trial.suggest_int("n_estimators", 50, 300),
        "max_depth": lambda trial: trial.suggest_int("max_depth", 3, 10),
        "learning_rate": lambda trial: trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": lambda trial: trial.suggest_float("subsample", 0.6, 1.0),
    },
}


def get_search_space(model_type: str) -> dict[str, Callable[[optuna.Trial], Any]]:
    """Resolve and return the hyperparameter search space for the given model type.

    Parameters
    ----------
    model_type : str
        The type of the model (e.g. "logistic_regression", "random_forest").

    Returns
    -------
    dict
        A dictionary mapping parameter names to parameter-suggesting functions.

    Raises
    ------
    ValueError
        If the model type is not supported in the search spaces.
    """
    resolved_type = model_type.lower()
    
    # Handle aliases
    if resolved_type in ["random_forest_classifier", "random_forest_regressor"]:
        resolved_type = "random_forest"

    if resolved_type not in SEARCH_SPACES:
        supported = sorted(SEARCH_SPACES.keys())
        raise ValueError(
            f"No search space defined for model type '{model_type}'. "
            f"Supported search spaces: {supported}."
        )

    return SEARCH_SPACES[resolved_type]
