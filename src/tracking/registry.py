"""
registry.py — ModelRegistry manages champion model promotion, serialization, and retrieval.
"""

import shutil
import json
import joblib
from pathlib import Path
from typing import Any
from src.tracking.tracker import ExperimentTracker
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Manages model artifact promotion and deployment champion tracking.

    Parameters
    ----------
    tracker : ExperimentTracker
        An instance of ExperimentTracker to link and update model states.
    registry_dir : str
        Directory where champion models and metadata are deployed.
    """

    def __init__(self, tracker: ExperimentTracker, registry_dir: str = "outputs/registry") -> None:
        self.tracker = tracker
        self.registry_dir = Path(registry_dir)
        logger.debug("Initializing ModelRegistry at directory: %s", self.registry_dir)
        try:
            self.registry_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error("Failed to create registry directory: %s", e)
            raise RuntimeError(f"Model registry initialization failed: {e}") from e

    def promote(self, run_id: str) -> None:
        """Promote a run model to champion.

        Copies the model joblib artifact to the registry, saves metadata, and updates DB.

        Parameters
        ----------
        run_id : str
            The ID of the run to promote.
        """
        logger.info("Promoting run ID %s to champion...", run_id)
        try:
            # Retrieve run details from tracker
            run = self.tracker.get_run(run_id)
            
            src_path = Path(run.model_path)
            if not src_path.exists():
                raise FileNotFoundError(
                    f"Model artifact not found at '{run.model_path}' for run '{run_id}'."
                )

            dest_path = self.registry_dir / "champion.joblib"
            logger.debug("Copying model artifact from %s to %s", src_path, dest_path)
            shutil.copy2(src_path, dest_path)

            # Copy the fitted pipeline engine alongside the model
            pipeline_src = Path(str(run.model_path).replace("model_", "pipeline_"))
            if pipeline_src.exists():
                pipeline_dest = self.registry_dir / "champion_pipeline.joblib"
                logger.debug("Copying pipeline artifact from %s to %s", pipeline_src, pipeline_dest)
                shutil.copy2(pipeline_src, pipeline_dest)
            else:
                logger.warning("No pipeline artifact found at '%s'; skipping pipeline promotion.", pipeline_src)

            # Update database champion flags
            self.tracker.set_champion(run_id)

            # Save champion metadata
            meta = {
                "run_id": run.run_id,
                "metrics": run.metrics,
                "config": run.config,
                "timestamp": run.timestamp,
            }
            meta_path = self.registry_dir / "champion_meta.json"
            logger.debug("Writing champion metadata to %s", meta_path)
            with meta_path.open("w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2)

            logger.info("Run ID %s successfully promoted to champion.", run_id)

        except Exception as e:
            logger.error("Failed to promote run %s to champion: %s", run_id, e)
            raise RuntimeError(f"Model promotion failed for run '{run_id}': {e}") from e

    def load_champion(self) -> tuple[Any, dict[str, Any]]:
        """Load and return the champion model artifact and metadata.

        Returns
        -------
        tuple[Any, dict]
            The loaded joblib model and the metadata dictionary.

        Raises
        ------
        FileNotFoundError
            If no champion model or metadata exists yet.
        """
        logger.debug("Loading champion model and metadata.")
        model_path = self.registry_dir / "champion.joblib"
        meta_path = self.registry_dir / "champion_meta.json"

        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError("No champion model promoted in the registry yet.")

        try:
            model = joblib.load(model_path)
            with meta_path.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)
            return model, meta
        except Exception as e:
            logger.error("Failed to load champion model/metadata: %s", e)
            raise RuntimeError(f"Error loading champion artifacts: {e}") from e

    def get_champion_meta(self) -> dict[str, Any]:
        """Retrieve the champion metadata without loading the model artifact.

        Returns
        -------
        dict
            The champion metadata dictionary.

        Raises
        ------
        FileNotFoundError
            If no champion metadata exists yet.
        """
        logger.debug("Retrieving champion metadata.")
        meta_path = self.registry_dir / "champion_meta.json"

        if not meta_path.exists():
            raise FileNotFoundError("No champion metadata found in the registry.")

        try:
            with meta_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as e:
            logger.error("Failed to load champion metadata: %s", e)
            raise RuntimeError(f"Error loading champion metadata: {e}") from e
