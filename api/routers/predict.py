"""
predict.py — API route for generating predictions using the champion model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import PredictRequest, PredictResponse
from src.tracking import ExperimentTracker, ModelRegistry

router = APIRouter(prefix="/api/predict", tags=["predict"])

REGISTRY_DIR = Path("outputs/registry")


def _load_champion():
    """Load the champion model and pipeline from the registry."""
    model_path = REGISTRY_DIR / "champion.joblib"
    pipeline_path = REGISTRY_DIR / "champion_pipeline.joblib"

    if not model_path.exists():
        raise FileNotFoundError("No champion model found. Promote a run first.")

    model = joblib.load(model_path)

    pipeline_engine = None
    if pipeline_path.exists():
        pipeline_engine = joblib.load(pipeline_path)

    return model, pipeline_engine


@router.post("/", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Generate predictions for the given data using the champion model."""
    try:
        model, pipeline_engine = _load_champion()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load champion: {e}")

    # Build DataFrame from request data
    try:
        df = pd.DataFrame(req.data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {e}")

    # Apply pipeline transformations if available
    if pipeline_engine is not None:
        try:
            df = pipeline_engine.transform(df)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Pipeline transformation failed: {e}",
            )

    # Generate predictions
    try:
        predictions = model.predict(df).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Get champion metadata for response
    tracker = ExperimentTracker(db_path="outputs/experiments.db")
    champion = tracker.get_champion()
    model_run_id = champion.run_id if champion else "unknown"
    model_type = champion.model_type if champion else "unknown"

    return PredictResponse(
        predictions=predictions,
        model_run_id=model_run_id,
        model_type=model_type,
    )
