"""
background.py — In-memory run status tracking + async background task executor.
"""

from __future__ import annotations

import sys
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

# Ensure project root is importable inside background tasks
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    from src.tracking import ExperimentTracker

# ── In-memory state ────────────────────────────────────────────────────────────
run_status: dict[str, str] = {}   # run_id → "running" | "complete" | "failed"
run_errors: dict[str, str] = {}   # run_id → error message


async def run_experiment_task(
    run_id: str,
    config: dict,
    tune: bool,
    n_trials: int,
    metric: str,
    tracker: "ExperimentTracker",
) -> None:
    """Execute the full ML pipeline in the background.

    Updates run_status[run_id] to "running", "complete", or "failed".
    Errors are captured in run_errors[run_id].
    """
    run_status[run_id] = "running"
    try:
        # Run the synchronous pipeline in a thread pool so we don't block the
        # event loop. We import run_pipeline here to avoid circular imports at
        # module load time.
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            _run_sync,
            run_id,
            config,
            tune,
            n_trials,
            metric,
        )
        run_status[run_id] = "complete"
    except Exception as exc:
        run_status[run_id] = "failed"
        run_errors[run_id] = str(exc)


def _run_sync(
    run_id: str,
    config: dict,
    tune: bool,
    n_trials: int,
    metric: str,
) -> None:
    """Synchronous wrapper called inside a thread executor."""
    from run import run_pipeline  # noqa: PLC0415

    run_pipeline(
        config=config,
        tune=tune,
        n_trials=n_trials,
        metric=metric,
        promote=False,
        tags={"api_run_id": run_id},
    )
