"""
pipeline.py — API routes for pipeline step validation and preview.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.engine import PipelineEngine

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


class PipelineValidationRequest(BaseModel):
    steps: list[dict[str, Any]]


class PipelinePreviewRequest(BaseModel):
    steps: list[dict[str, Any]]
    data_path: str
    n_rows: int = 20


@router.post("/validate")
async def validate_pipeline(req: PipelineValidationRequest):
    """Validate that the pipeline step definitions are syntactically correct."""
    try:
        engine = PipelineEngine(req.steps)
        summary = engine.get_steps_summary()
        return {
            "valid": True,
            "steps_count": len(req.steps),
            "summary": summary,
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
        }


@router.post("/preview")
async def preview_pipeline(req: PipelinePreviewRequest):
    """Fit-transform a subset of data through the pipeline to preview output."""
    import pandas as pd

    data_path = Path(req.data_path)
    if not data_path.exists():
        raise HTTPException(status_code=404, detail=f"Data file '{req.data_path}' not found.")

    try:
        df = pd.read_csv(data_path, nrows=req.n_rows)
        engine = PipelineEngine(req.steps)
        transformed = engine.fit_transform(df)
        return {
            "columns_before": list(df.columns),
            "columns_after": list(transformed.columns),
            "rows_before": len(df),
            "rows_after": len(transformed),
            "preview": transformed.head(req.n_rows).to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Pipeline preview failed: {e}")
