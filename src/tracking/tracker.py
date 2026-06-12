"""
tracker.py — ExperimentTracker logs and manages RunRecords in an SQLite database.
"""

import sqlite3
import json
from pathlib import Path
from typing import Any
from src.utils.logger import get_logger
from src.tracking.run_record import RunRecord

logger = get_logger(__name__)


class ExperimentTracker:
    """Manages tracking records in an SQLite database.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    """

    def __init__(self, db_path: str = "outputs/experiments.db") -> None:
        if str(db_path) == ":memory:":
            self.db_path = ":memory:"
            self._conn = sqlite3.connect(":memory:")
            self._conn.row_factory = sqlite3.Row
        else:
            self.db_path = Path(db_path)
            self._conn = None
            try:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error("Failed to create database directories: %s", e)
                raise RuntimeError(f"Database directory creation failed: {e}") from e

        logger.debug("Initializing ExperimentTracker at database path: %s", self.db_path)
        try:
            self._init_db()
        except Exception as e:
            logger.error("Failed to initialize database table: %s", e)
            raise RuntimeError(f"Database initialization failed: {e}") from e

    def _get_connection(self):
        """Helper to get a database connection. Returns a connection context manager.

        If in-memory, returns a dummy context wrapping the persistent connection.
        Otherwise, returns a new sqlite3 connection.
        """
        if self._conn is not None:
            class DummyContext:
                def __init__(self, conn):
                    self.conn = conn
                def __enter__(self):
                    return self.conn
                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass
            return DummyContext(self._conn)
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn

    def _init_db(self) -> None:
        """Create the runs table if it does not already exist."""
        logger.debug("Ensuring runs table exists in DB.")
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    config TEXT,
                    pipeline_summary TEXT,
                    metrics TEXT,
                    model_type TEXT,
                    model_path TEXT,
                    dataset_path TEXT,
                    train_rows INTEGER,
                    test_rows INTEGER,
                    duration_seconds REAL,
                    tags TEXT,
                    is_champion INTEGER DEFAULT 0
                )
                """
            )
            conn.commit()

    def log_run(self, record: RunRecord) -> str:
        """Insert or replace a RunRecord into the database.

        Parameters
        ----------
        record : RunRecord
            The run record to persist.

        Returns
        -------
        str
            The run ID of the logged run.
        """
        logger.debug("Logging run record %s to database.", record.run_id)
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO runs (
                        run_id, timestamp, config, pipeline_summary, metrics,
                        model_type, model_path, dataset_path, train_rows, test_rows,
                        duration_seconds, tags, is_champion
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.run_id,
                        record.timestamp,
                        json.dumps(record.config),
                        json.dumps(record.pipeline_summary),
                        json.dumps(record.metrics),
                        record.model_type,
                        record.model_path,
                        record.dataset_path,
                        record.train_rows,
                        record.test_rows,
                        record.duration_seconds,
                        json.dumps(record.tags),
                        1 if record.is_champion else 0,
                    ),
                )
                conn.commit()
            return record.run_id
        except Exception as e:
            logger.error("Error logging run %s: %s", record.run_id, e)
            raise RuntimeError(f"Failed to log run record in DB: {e}") from e

    def get_run(self, run_id: str) -> RunRecord:
        """Retrieve a single RunRecord from the database by ID or 8-character short ID prefix.

        Parameters
        ----------
        run_id : str
            The ID or short ID prefix of the run to fetch.

        Returns
        -------
        RunRecord
            The reconstructed RunRecord.

        Raises
        ------
        ValueError
            If the run ID is not found or is ambiguous.
        """
        logger.debug("Retrieving run record %s from database.", run_id)
        try:
            with self._get_connection() as conn:
                if len(run_id) == 8:
                    cursor = conn.execute(
                        "SELECT * FROM runs WHERE run_id LIKE ?", (f"{run_id}%",)
                    )
                    rows = cursor.fetchall()
                    if not rows:
                        raise ValueError(f"Run ID prefix '{run_id}' not found in database.")
                    if len(rows) > 1:
                        matched = [r["run_id"] for r in rows]
                        raise ValueError(
                            f"Ambiguous Run ID prefix '{run_id}' matched multiple runs: {matched}"
                        )
                    row = rows[0]
                else:
                    cursor = conn.execute(
                        "SELECT * FROM runs WHERE run_id = ?", (run_id,)
                    )
                    row = cursor.fetchone()
                    if row is None:
                        raise ValueError(f"Run ID '{run_id}' not found in database.")
                return RunRecord.from_dict(dict(row))
        except ValueError:
            raise
        except Exception as e:
            logger.error("Error retrieving run %s: %s", run_id, e)
            raise RuntimeError(f"Failed to query database for run ID '{run_id}': {e}") from e

    def list_runs(self, limit: int = 50) -> list[RunRecord]:
        """List past runs sorted by timestamp descending.

        Parameters
        ----------
        limit : int
            Maximum number of runs to return.

        Returns
        -------
        list[RunRecord]
            List of runs.
        """
        logger.debug("Listing up to %d runs.", limit)
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (limit,)
                )
                rows = cursor.fetchall()
                return [RunRecord.from_dict(dict(row)) for row in rows]
        except Exception as e:
            logger.error("Error listing runs: %s", e)
            raise RuntimeError(f"Failed to list run records from DB: {e}") from e

    def delete_run(self, run_id: str) -> None:
        """Delete a run from the database. Supports 8-char short ID prefixes.

        Parameters
        ----------
        run_id : str
            The ID of the run to delete.
        """
        logger.debug("Deleting run record %s from database.", run_id)
        try:
            # Resolve full run ID if given a prefix
            resolved = self.get_run(run_id)
            full_id = resolved.run_id

            with self._get_connection() as conn:
                cursor = conn.execute("DELETE FROM runs WHERE run_id = ?", (full_id,))
                conn.commit()
        except Exception as e:
            logger.error("Error deleting run %s: %s", run_id, e)
            raise RuntimeError(f"Failed to delete run ID '{run_id}' from DB: {e}") from e

    def set_champion(self, run_id: str) -> None:
        """Mark a run as champion, clearing all other champions. Supports 8-char short ID prefixes.

        Parameters
        ----------
        run_id : str
            The ID of the run to promote to champion.
        """
        logger.debug("Promoting run ID %s to champion status.", run_id)
        try:
            # Resolve full run ID if given a prefix
            resolved = self.get_run(run_id)
            full_id = resolved.run_id
            
            with self._get_connection() as conn:
                conn.execute("UPDATE runs SET is_champion = 0")
                conn.execute("UPDATE runs SET is_champion = 1 WHERE run_id = ?", (full_id,))
                conn.commit()
            logger.debug("Successfully updated champion status in database for run %s.", full_id)
        except Exception as e:
            logger.error("Error setting champion to %s: %s", run_id, e)
            raise RuntimeError(f"Failed to promote run ID '{run_id}' to champion: {e}") from e

    def get_champion(self) -> RunRecord | None:
        """Retrieve the current champion run.

        Returns
        -------
        RunRecord | None
            The champion RunRecord, or None if no champion has been set.
        """
        logger.debug("Fetching the champion run record.")
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT * FROM runs WHERE is_champion = 1")
                row = cursor.fetchone()
                if row is None:
                    logger.debug("No champion run record found.")
                    return None
                return RunRecord.from_dict(dict(row))
        except Exception as e:
            logger.error("Error fetching champion run: %s", e)
            raise RuntimeError(f"Failed to fetch champion run record: {e}") from e
