"""
logger.py — Configures a dual-output (console + file) Python logger.

Usage
-----
>>> from src.utils.logger import get_logger
>>> log = get_logger(__name__)
>>> log.info("Hello from %s", __name__)
"""

import logging
import sys
from pathlib import Path

# Single log file written to the project root — all modules share it.
_LOG_FILE = Path("run.log")

# Guard against adding duplicate handlers when the same logger is requested
# multiple times within one interpreter session.
_configured_loggers: set[str] = set()

_FMT = "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with console and file handlers attached.

    The logger format is::

        [TIMESTAMP] [LEVEL   ] [module_name] message

    Both the console (stdout) and ``run.log`` in the current working
    directory receive all messages at ``DEBUG`` level and above.

    Parameters
    ----------
    name : str
        Logger name — typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        A fully configured :class:`logging.Logger` instance.

    Notes
    -----
    Calling this function multiple times with the same *name* is safe;
    handlers are only attached once.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if name in _configured_loggers:
        return logger

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt=_FMT, datefmt=_DATE_FMT)

    # ── Console handler (stdout) ─────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # ── File handler ─────────────────────────────────────────────────────
    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False  # Prevent double-logging via root logger

    _configured_loggers.add(name)
    return logger
