"""Logging utilities for Daria-Former."""

from __future__ import annotations

import logging
import sys


_configured = False


def setup_logging(level: int = logging.INFO):
    """Configure root logger for Daria-Former."""
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger("daria_former")
    root.setLevel(level)
    root.addHandler(handler)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given module name."""
    setup_logging()
    return logging.getLogger(name)
