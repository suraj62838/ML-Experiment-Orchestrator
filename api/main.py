"""
main.py — FastAPI application entry point for the ML Experiment Orchestrator.

Run with:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.routers import experiments, datasets, pipeline, predict, drift

app = FastAPI(
    title="OrchestrML API",
    description="REST API for the ML Experiment Orchestrator — manage experiments, predict, and monitor drift.",
    version="1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Mount routers ─────────────────────────────────────────────────────────────
app.include_router(experiments.router)
app.include_router(datasets.router)
app.include_router(pipeline.router)
app.include_router(predict.router)
app.include_router(drift.router)


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    """Ensure required directories exist."""
    for d in ["outputs", "uploads", "outputs/registry"]:
        Path(d).mkdir(parents=True, exist_ok=True)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "OrchestrML API"}
