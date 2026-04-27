"""Shared logging helpers for command-line scripts."""

from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a simple logging format for project scripts."""

    logging.basicConfig(
        level=level,
        format="%(levelname)s | %(name)s | %(message)s",
    )
