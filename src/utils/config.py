"""
config.py — Reads YAML experiment configuration files into Python dicts.

Usage
-----
>>> from src.utils.config import load_config
>>> cfg = load_config("configs/experiment.yaml")
>>> print(cfg["model"]["type"])
'logistic_regression'
"""

from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    """Read a YAML file and return its contents as a Python dict.

    Parameters
    ----------
    path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given *path*.
    ValueError
        If the file cannot be parsed as valid YAML, or if the parsed
        content is not a mapping (dict).
    """
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: '{path}'. "
            "Provide a valid path with --config."
        )

    try:
        with config_path.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Failed to parse YAML configuration '{path}': {exc}"
        ) from exc

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration file '{path}' must contain a YAML mapping (dict) "
            f"at the top level, but got: {type(config).__name__}."
        )

    return config
