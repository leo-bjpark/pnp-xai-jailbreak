"""Load config.yaml and expose XAI_LEVEL_NAMES."""

from pathlib import Path
from typing import Dict, List, Tuple

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _load_xai_level_list() -> List[Tuple[str, str]]:
    """
    Parse XAI_LEVEL_NAMES from config as ordered list of (level, name).
    Preserves config order and duplicate level keys (e.g. two 0.1 entries).
    """
    if not CONFIG_PATH.exists():
        return [
            ("0.1", "Positive and Negative Response Preference"),
            ("1.1", "Response Attribution"),
        ]

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw = data.get("XAI_LEVEL_NAMES")
    if not raw:
        return []

    result: List[Tuple[str, str]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                for k, v in item.items():
                    result.append((str(k).strip(), str(v).strip()))
    elif isinstance(raw, dict):
        for k, v in raw.items():
            result.append((str(k).strip(), str(v).strip()))
    return result


def get_xai_level_names_grouped() -> Dict[int, List[Tuple[str, str]]]:
    """
    Return levels grouped by integer part from config list order.
    E.g. {0: [(0.1, "Simple Generation"), (0.1, "Positive and ..."), (0.2, "Binary ..."), ...], 1: [...], ...}
    """
    ordered = _load_xai_level_list()
    grouped: Dict[int, List[Tuple[str, str]]] = {}
    for level_str, name in ordered:
        try:
            major = int(float(level_str))
        except (ValueError, TypeError):
            major = 0
        if major not in grouped:
            grouped[major] = []
        grouped[major].append((level_str, name))
    return dict(sorted(grouped.items()))


def get_xai_level_names() -> Dict[str, str]:
    """
    Return XAI level -> name mapping (last occurrence per level for display).
    Built from config list so all levels from config are included.
    """
    ordered = _load_xai_level_list()
    return dict(ordered)
