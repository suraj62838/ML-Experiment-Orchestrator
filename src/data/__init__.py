"""
Data subpackage — handles all data pipeline stages:
  - DataLoader:   Reads CSV, JSON, Parquet files into DataFrames
  - DataCleaner:  Applies configurable cleaning steps
  - DataSplitter: Splits data into train/val/test sets
"""

from .loader import DataLoader
from .cleaner import DataCleaner
from .splitter import DataSplitter

__all__ = ["DataLoader", "DataCleaner", "DataSplitter"]
