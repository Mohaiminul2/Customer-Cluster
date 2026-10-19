"""Shared configuration loading for the Customer Segmentation project."""

from __future__ import annotations

import os
from typing import Any

import yaml

BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load config.yaml from the project root."""
    if path is None:
        path = os.path.join(BASE_DIR, "config.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
