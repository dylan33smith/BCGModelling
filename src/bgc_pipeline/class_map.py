"""Load harmonised COMPOUND_CLASS labels from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_class_map(path: Path) -> tuple[dict[str, str], str]:
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = data.get("mappings") or {}
    normalized = {str(k).strip().lower(): str(v).strip() for k, v in raw.items()}
    default = str(data.get("default_class", "OTHER")).strip()
    return normalized, default


def map_mibig_class(mibig_class: str, mapping: dict[str, str], default: str) -> str:
    key = (mibig_class or "").strip().lower()
    return mapping.get(key, default)
