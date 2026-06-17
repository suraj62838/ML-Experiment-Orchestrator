"""
drift.py — API route for detecting data drift against the champion model's
training baseline.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.drift import DriftDetector
from src.tracking import ExperimentTracker
from api.schemas import DriftReport

router = APIRouter(prefix="/api/drift", tags=["drift"])

DRIFT_REPORT_PATH = Path("outputs/drift_report.json")


def _get_reference_df() -> pd.DataFrame:
    """Load the training dataset that was used to train the current champion model."""
    tracker = ExperimentTracker(db_path="outputs/experiments.db")
    champion = tracker.get_champion()
    if champion is None:
        raise FileNotFoundError("No champion model found. Promote a run first.")

    dataset_path = Path(champion.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Champion training dataset not found at '{dataset_path}'.")

    return pd.read_csv(dataset_path)


@router.post("/detect")
async def detect_drift(data_path: str):
    """Run drift detection on the dataset at the given path vs. the champion baseline."""
    new_path = Path(data_path)
    if not new_path.exists():
        raise HTTPException(status_code=404, detail=f"Data file '{data_path}' not found.")

    try:
        reference_df = _get_reference_df()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    new_df = pd.read_csv(new_path)

    detector = DriftDetector(reference_df)
    report = detector.detect(new_df)

    # Persist the latest report
    DRIFT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DRIFT_REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


@router.get("/latest", response_model=DriftReport)
async def get_latest_drift_report():
    """Return the most recent drift report."""
    if not DRIFT_REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="No drift report available yet.")

    with DRIFT_REPORT_PATH.open("r", encoding="utf-8") as f:
        report = json.load(f)

    return report
