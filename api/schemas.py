"""
schemas.py — Pydantic models for the OrchestrML API request/response contracts.
"""

from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


# ── Request schemas ────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    config: dict[str, Any]
    tune: bool = False
    n_trials: int = 20
    metric: str = "f1"
    tags: dict[str, Any] = Field(default_factory=dict)


class PredictRequest(BaseModel):
    data: list[dict[str, Any]]


# ── Response schemas ───────────────────────────────────────────────────────────

class RunResponse(BaseModel):
    run_id: str
    status: str
    message: str


class ExperimentSummary(BaseModel):
    run_id: str
    timestamp: str
    model_type: str
    metrics: dict[str, Any]
    is_champion: bool
    duration_seconds: float
    tags: dict[str, Any]
    pipeline_summary: list[Any]


class PredictResponse(BaseModel):
    predictions: list[Any]
    model_run_id: str
    model_type: str


class DriftReport(BaseModel):
    status: str                  # "ok" | "warning" | "alert"
    features_drifted: list[str]
    drift_scores: dict[str, Any]
    timestamp: str
