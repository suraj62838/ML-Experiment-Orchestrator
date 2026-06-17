"""
datasets.py — API routes for dataset upload, listing, preview, and deletion.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/")
async def list_datasets():
    """List all uploaded dataset files."""
    files = []
    for f in UPLOAD_DIR.iterdir():
        if f.is_file() and f.suffix == ".csv":
            files.append({
                "name": f.name,
                "size_bytes": f.stat().st_size,
                "path": str(f),
            })
    return {"datasets": files}


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV dataset file."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    dest = UPLOAD_DIR / file.filename
    contents = await file.read()
    dest.write_bytes(contents)

    return {
        "message": f"File '{file.filename}' uploaded successfully.",
        "path": str(dest),
        "size_bytes": len(contents),
    }


@router.get("/preview/{filename}")
async def preview_dataset(filename: str, rows: int = 20):
    """Preview the first N rows of an uploaded CSV dataset."""
    filepath = UPLOAD_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")

    import pandas as pd
    try:
        df = pd.read_csv(filepath, nrows=rows)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    return {
        "filename": filename,
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
        "total_preview": len(df),
    }


@router.delete("/{filename}")
async def delete_dataset(filename: str):
    """Delete an uploaded dataset file."""
    filepath = UPLOAD_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")

    os.remove(filepath)
    return {"message": f"File '{filename}' deleted."}
