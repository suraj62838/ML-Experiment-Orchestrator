"""
tuner.py — HyperparameterTuner runs hyperparameter search trials with Optuna.
"""

import time
import uuid
import optuna
import pandas as pd
from tqdm import tqdm
from typing import Any

from src.utils.logger import get_logger
from src.models.trainer import ModelTrainer
from src.models.evaluator import Evaluator
from src.tracking.tracker import ExperimentTracker
from src.tracking.run_record import RunRecord
from src.tuning.search_spaces import get_search_space

logger = get_logger(__name__)

# Suppress Optuna default output logging to maintain custom orchestrator logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterTuner:
    """Orchestrates Optuna trials for model hyperparameter optimization.

    Parameters
    ----------
    config : dict
        Base configuration mapping (must contain "model").
    n_trials : int
        Number of search trials to run.
    metric : str
        The metric to optimize (e.g. "f1", "rmse", "accuracy").
    direction : str
        Optimization direction ("maximize" or "minimize").
    tracker : ExperimentTracker, optional
        Tracker instance to log each trial's execution metadata.
    """

    def __init__(
        self,
        config: dict,
        n_trials: int = 20,
        metric: str = "f1",
        direction: str = None,
        tracker: ExperimentTracker = None,
    ) -> None:
        self.config = config
        self.n_trials = n_trials
        self.metric = metric
        self.tracker = tracker
        self.study = None
        self.trial_records: dict[int, RunRecord] = {}

        # Default direction if not provided
        if direction is not None:
            self.direction = direction
        else:
            self.direction = "minimize" if metric.lower() in ["rmse", "mae"] else "maximize"

        logger.debug(
            "Initialized HyperparameterTuner: trials=%d, metric=%s, direction=%s",
            self.n_trials,
            self.metric,
            self.direction,
        )

    def _get_metric_value(self, metrics: dict[str, Any]) -> float:
        """Extract target metric value from Evaluator dictionary, resolving aliases."""
        if self.metric in metrics:
            return float(metrics[self.metric])

        # Match common metric aliases
        aliases = {
            "f1": "f1_weighted",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
        }
        mapped_key = aliases.get(self.metric.lower())
        if mapped_key and mapped_key in metrics:
            return float(metrics[mapped_key])

        raise ValueError(
            f"Tuner target metric '{self.metric}' not found in evaluated metrics: "
            f"{list(metrics.keys())}"
        )

    def tune(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        pipeline_summary: list = None,
    ) -> RunRecord:
        """Run hyperparameter search optimization over validation data.

        Parameters
        ----------
        X_train : array-like
            Training feature matrix.
        y_train : array-like
            Training labels.
        X_val : array-like
            Validation feature matrix.
        y_val : array-like
            Validation labels.
        pipeline_summary : list, optional
            List of dictionary structures describing the preprocessing steps.

        Returns
        -------
        RunRecord
            The RunRecord associated with the best trial.
        """
        logger.info(
            "Starting hyperparameter tuning study (%d trials, optimizing '%s' via %s)...",
            self.n_trials,
            self.metric,
            self.direction,
        )

        model_type = self.config["model"]["type"]
        search_space = get_search_space(model_type)
        evaluator = Evaluator()

        self.study = optuna.create_study(direction=self.direction)
        self.trial_records = {}

        def objective(trial: optuna.Trial) -> float:
            trial_start_time = time.time()
            trial_run_id = str(uuid.uuid4())

            # Sample parameters from search space
            params = {name: suggest_fn(trial) for name, suggest_fn in search_space.items()}
            logger.debug("Trial %d: Sampled params = %s", trial.number, params)

            # Fit and evaluate
            try:
                trainer = ModelTrainer(model_type=model_type, model_params=params)
                trainer.fit(X_train, y_train)
                
                # Save trial model artifact
                model_path = f"outputs/model_{trial_run_id}.joblib"
                trainer.save(model_path)
                
                metrics = evaluator.evaluate(trainer.model, X_val, y_val)
                score = self._get_metric_value(metrics)
            except Exception as e:
                logger.error("Trial %d failed with error: %s", trial.number, e)
                raise

            duration = time.time() - trial_start_time

            # Update base config to match trial params
            trial_config = self.config.copy()
            trial_config["model"] = trial_config.get("model", {}).copy()
            trial_config["model"]["params"] = params

            # Construct RunRecord for this trial
            trial_tags = {
                "type": "trial",
                "trial_number": str(trial.number),
            }
            if "tags" in self.config:
                trial_tags.update(self.config["tags"])

            record = RunRecord(
                run_id=trial_run_id,
                config=trial_config,
                pipeline_summary=pipeline_summary or [],
                metrics=metrics,
                model_type=model_type,
                model_path=model_path,
                dataset_path=self.config["data"]["filepath"],
                train_rows=len(X_train),
                test_rows=len(X_val),
                duration_seconds=duration,
                tags=trial_tags,
                is_champion=False,
            )

            # Save the record locally
            self.trial_records[trial.number] = record

            # Log to SQLite database if tracker is available
            if self.tracker:
                self.tracker.log_run(record)

            return score

        # Run optimization loops wrapped in tqdm progress bar
        for _ in tqdm(range(self.n_trials), desc="Optuna Tuning Trials"):
            self.study.optimize(objective, n_trials=1)

        best_trial = self.study.best_trial
        logger.info(
            "Tuning study completed. Best trial: #%d, value: %s, params: %s",
            best_trial.number,
            best_trial.value,
            best_trial.params,
        )

        return self.trial_records[best_trial.number]

    def get_study_summary(self) -> pd.DataFrame:
        """Return a summary of all completed trials as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame of all trials.
        """
        if self.study is None:
            raise RuntimeError("No tuning study has been run yet. Call tune() first.")

        records = []
        for trial in self.study.trials:
            if trial.state.name != "COMPLETE":
                continue
            row = {
                "trial": trial.number,
                "value": trial.value,
            }
            row.update(trial.params)
            records.append(row)

        return pd.DataFrame(records)
