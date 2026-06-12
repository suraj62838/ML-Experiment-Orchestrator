"""
run_record.py — Dataclass structure representing a single experiment run.
"""

import uuid
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunRecord:
    """Represents the complete configuration, execution, and evaluation metadata of a single run."""
    config: dict
    pipeline_summary: list
    metrics: dict
    model_type: str
    model_path: str
    dataset_path: str
    train_rows: int
    test_rows: int
    duration_seconds: float
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: dict = field(default_factory=dict)
    is_champion: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert the RunRecord to a dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "config": self.config,
            "pipeline_summary": self.pipeline_summary,
            "metrics": self.metrics,
            "model_type": self.model_type,
            "model_path": self.model_path,
            "dataset_path": self.dataset_path,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "duration_seconds": self.duration_seconds,
            "tags": self.tags,
            "is_champion": self.is_champion,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunRecord":
        """Reconstruct a RunRecord from a dictionary, parsing JSON fields if they are string serialized."""
        
        def parse_json(field_val: Any) -> Any:
            if isinstance(field_val, str):
                try:
                    return json.loads(field_val)
                except json.JSONDecodeError:
                    return field_val
            return field_val

        config = parse_json(d.get("config", {}))
        pipeline_summary = parse_json(d.get("pipeline_summary", []))
        metrics = parse_json(d.get("metrics", {}))
        tags = parse_json(d.get("tags", {}))
        
        is_champ = d.get("is_champion", False)
        if isinstance(is_champ, int):
            is_champ = bool(is_champ)

        return cls(
            run_id=str(d.get("run_id") or str(uuid.uuid4())),
            timestamp=str(d.get("timestamp") or datetime.now().isoformat()),
            config=config,
            pipeline_summary=pipeline_summary,
            metrics=metrics,
            model_type=str(d.get("model_type", "")),
            model_path=str(d.get("model_path", "")),
            dataset_path=str(d.get("dataset_path", "")),
            train_rows=int(d.get("train_rows", 0)),
            test_rows=int(d.get("test_rows", 0)),
            duration_seconds=float(d.get("duration_seconds", 0.0)),
            tags=tags,
            is_champion=is_champ,
        )
