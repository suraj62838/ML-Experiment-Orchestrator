"""
experiments.py — API routes for experiment run management.
"""

from __future__ import annotations

import sys
import uuid
import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking import ExperimentTracker, ModelRegistry
from src.utils.config import load_config
from api.schemas import RunRequest, RunResponse, ExperimentSummary
from api.background import run_experiment_task, run_status, run_errors

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


def _get_tracker() -> ExperimentTracker:
    return ExperimentTracker(db_path="outputs/experiments.db")


def _record_to_summary(rec) -> dict[str, Any]:
    """Convert a RunRecord into an ExperimentSummary-compatible dict."""
    return {
        "run_id": rec.run_id,
        "timestamp": rec.timestamp,
        "model_type": rec.model_type,
        "metrics": rec.metrics,
        "is_champion": rec.is_champion,
        "duration_seconds": rec.duration_seconds,
        "tags": rec.tags,
        "pipeline_summary": rec.pipeline_summary,
    }


# ── List all runs ─────────────────────────────────────────────────────────────

@router.get("/", response_model=list[ExperimentSummary])
async def list_experiments(limit: int = Query(50, ge=1, le=200)):
    """Return the most recent experiment runs."""
    tracker = _get_tracker()
    runs = tracker.list_runs(limit=limit)
    return [_record_to_summary(r) for r in runs]


# ── Get single run detail ────────────────────────────────────────────────────

@router.get("/{run_id}")
async def get_experiment(run_id: str):
    """Fetch full details for a specific run."""
    tracker = _get_tracker()
    try:
        rec = tracker.get_run(run_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return rec.to_dict()


# ── Submit a new run ──────────────────────────────────────────────────────────

@router.post("/", response_model=RunResponse)
async def submit_run(req: RunRequest):
    """Launch a new experiment run in the background."""
    run_id = str(uuid.uuid4())

    # Merge user-supplied config with the default experiment.yaml
    default_cfg_path = PROJECT_ROOT / "configs" / "experiment.yaml"
    if default_cfg_path.exists():
        config = load_config(str(default_cfg_path))
    else:
        config = {}

    # Override with any user-specified config sections
    for key, val in req.config.items():
        config[key] = val

    tracker = _get_tracker()

    asyncio.create_task(
        run_experiment_task(
            run_id=run_id,
            config=config,
            tune=req.tune,
            n_trials=req.n_trials,
            metric=req.metric,
            tracker=tracker,
        )
    )

    return RunResponse(
        run_id=run_id,
        status="submitted",
        message="Experiment queued for execution.",
    )


# ── Check run status ─────────────────────────────────────────────────────────

@router.get("/{run_id}/status")
async def get_run_status(run_id: str):
    """Check the status of a background run."""
    status = run_status.get(run_id, "unknown")
    error = run_errors.get(run_id)
    return {"run_id": run_id, "status": status, "error": error}


# ── Delete a run ──────────────────────────────────────────────────────────────

@router.delete("/{run_id}")
async def delete_experiment(run_id: str):
    """Delete an experiment run from the database."""
    tracker = _get_tracker()
    try:
        tracker.delete_run(run_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"message": f"Run {run_id} deleted."}


# ── Promote to champion ──────────────────────────────────────────────────────

@router.post("/{run_id}/promote")
async def promote_run(run_id: str):
    """Promote a run to champion status."""
    tracker = _get_tracker()
    registry = ModelRegistry(tracker=tracker)
    try:
        registry.promote(run_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"message": f"Run {run_id} promoted to champion."}


# ── Compare two runs ─────────────────────────────────────────────────────────

@router.get("/compare/{run_id_a}/{run_id_b}")
async def compare_runs(run_id_a: str, run_id_b: str):
    """Side-by-side comparison of two experiment runs."""
    tracker = _get_tracker()
    try:
        rec_a = tracker.get_run(run_id_a)
        rec_b = tracker.get_run(run_id_b)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"run_a": rec_a.to_dict(), "run_b": rec_b.to_dict()}
