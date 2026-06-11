"""
test_loader.py — Unit tests for DataLoader.

All tests use in-memory data or temporary files (no network or real FS deps).
Run with: pytest tests/
"""

import io
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# ── path fix so tests can find `src` without installing the package ──────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """A small DataFrame used across multiple tests."""
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "target": [0, 1, 0]})


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataLoaderCSV:
    """Tests for CSV loading."""

    def test_load_csv_returns_dataframe(self, sample_df, tmp_path):
        """load() should return a DataFrame with the same shape as the source."""
        csv_path = tmp_path / "test.csv"
        sample_df.to_csv(csv_path, index=False)

        loader = DataLoader()
        result = loader.load(str(csv_path))

        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_df.shape

    def test_load_csv_column_names(self, sample_df, tmp_path):
        """Column names must be preserved after loading."""
        csv_path = tmp_path / "test.csv"
        sample_df.to_csv(csv_path, index=False)

        loader = DataLoader()
        result = loader.load(str(csv_path))

        assert list(result.columns) == list(sample_df.columns)

    def test_load_csv_values(self, sample_df, tmp_path):
        """Numeric values must round-trip correctly through CSV."""
        csv_path = tmp_path / "test.csv"
        sample_df.to_csv(csv_path, index=False)

        loader = DataLoader()
        result = loader.load(str(csv_path))

        pd.testing.assert_frame_equal(result.reset_index(drop=True), sample_df.reset_index(drop=True))


class TestDataLoaderJSON:
    """Tests for JSON loading."""

    def test_load_json_returns_dataframe(self, sample_df, tmp_path):
        """load() should handle JSON files and return a DataFrame."""
        json_path = tmp_path / "test.json"
        sample_df.to_json(json_path, orient="records")

        loader = DataLoader()
        result = loader.load(str(json_path))

        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_df.shape

    def test_load_json_columns(self, sample_df, tmp_path):
        """Column names must survive JSON serialisation."""
        json_path = tmp_path / "test.json"
        sample_df.to_json(json_path, orient="records")

        loader = DataLoader()
        result = loader.load(str(json_path))

        assert set(result.columns) == set(sample_df.columns)


class TestDataLoaderErrors:
    """Tests for error handling in DataLoader."""

    def test_raises_file_not_found(self, tmp_path):
        """A FileNotFoundError is raised when the file does not exist."""
        loader = DataLoader()
        with pytest.raises(FileNotFoundError, match="not found"):
            loader.load(str(tmp_path / "nonexistent.csv"))

    def test_raises_unsupported_format(self, tmp_path):
        """A ValueError is raised for unknown file extensions."""
        bad_file = tmp_path / "data.xlsx"
        bad_file.write_text("dummy")

        loader = DataLoader()
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load(str(bad_file))

    def test_raises_on_txt_extension(self, tmp_path):
        """Plain .txt files are not supported."""
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("a,b\n1,2")

        loader = DataLoader()
        with pytest.raises(ValueError):
            loader.load(str(txt_file))
