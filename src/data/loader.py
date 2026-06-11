"""
loader.py — Reads tabular data files into pandas DataFrames.

Supported formats: CSV, JSON, Parquet.
The format is auto-detected from the file extension.
"""

import pandas as pd
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Map file extension → pandas reader function
_READERS: dict[str, callable] = {
    ".csv": pd.read_csv,
    ".json": pd.read_json,
    ".parquet": pd.read_parquet,
}


class DataLoader:
    """Loads a dataset from disk into a pandas DataFrame.

    The reader is chosen automatically based on the file extension.

    Example
    -------
    >>> loader = DataLoader()
    >>> df = loader.load("data/sample.csv")
    """

    def load(self, filepath: str) -> pd.DataFrame:
        """Load a file and return it as a DataFrame.

        Parameters
        ----------
        filepath : str
            Absolute or relative path to the data file.
            Supported extensions: ``.csv``, ``.json``, ``.parquet``.

        Returns
        -------
        pd.DataFrame
            The loaded data.

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist on disk.
        ValueError
            If the file extension is not supported.
        """
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(
                f"Data file not found: '{filepath}'. "
                "Check the 'data.filepath' field in your config."
            )

        ext = path.suffix.lower()
        if ext not in _READERS:
            supported = ", ".join(_READERS.keys())
            raise ValueError(
                f"Unsupported file format '{ext}'. "
                f"Supported formats: {supported}."
            )

        logger.info("Loading data from '%s' (format: %s)…", filepath, ext)
        df: pd.DataFrame = _READERS[ext](path)

        logger.info(
            "Loaded '%s' — shape: %s | columns: %s",
            path.name,
            df.shape,
            df.dtypes.to_dict(),
        )
        return df
