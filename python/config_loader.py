"""Load config.yaml and expose XAI_LEVEL_NAMES."""

from pathlib import Path
from typing import Dict, List, Tuple

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

_KNOWN_NAME_TO_LEVEL = {
    "completion": "0.1.1",
    "conversation": "0.1.2",
    "binary classification": "0.2.1",
    "multiclass classification": "0.2.2",
    "regression": "0.2.3",
    "jailbreak": "0.2.4",
    "positive and negative response preference": "0.2.5",
    "response attribution": "1.0.1",
    "positive & negative attribution": "1.0.1",
    "positive and negative attribution": "1.0.1",
    "residual concept detection": "2.0.1",
    "layer direction similarity analysis": "2.0.2",
    "brain concept visualization": "2.1.0",
}


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
    used_levels = set()

    def _assign_level_id(name: str, major: int, idx: int) -> str:
        key = (name or "").strip().lower()
        mapped = _KNOWN_NAME_TO_LEVEL.get(key)
        if mapped and mapped not in used_levels:
            return mapped
        # fallback: major.<idx+1>, ensure uniqueness
        base = f"{major}.{idx + 1}"
        candidate = base
        suffix = 1
        while candidate in used_levels:
            suffix += 1
            candidate = f"{base}.{suffix}"
        return candidate

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            for k, v in item.items():
                key = str(k).strip()
                if key.upper().startswith("LEVEL_") and isinstance(v, list):
                    try:
                        major = int(key.split("_", 1)[1])
                    except (ValueError, IndexError):
                        major = 0
                    for idx, name in enumerate(v):
                        label = str(name).strip()
                        if not label:
                            continue
                        level_id = _assign_level_id(label, major, idx)
                        used_levels.add(level_id)
                        result.append((level_id, label))
                else:
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
        major = 0
        try:
            parts = str(level_str).strip().split(".")
            if parts and parts[0].isdigit():
                major = int(parts[0])
            else:
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


def get_dataset_groups() -> Dict[str, List[str]]:
    """Return dataset groups from config.yaml DATASETS section."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("DATASETS") or {}
    if isinstance(raw, dict):
        out: Dict[str, List[str]] = {}
        for k, v in raw.items():
            if isinstance(v, list):
                out[str(k).strip()] = [str(x).strip() for x in v if str(x).strip()]
        return out
    return {}


def get_dataset_list() -> List[str]:
    """Flattened dataset list from config.yaml DATASETS section."""
    groups = get_dataset_groups()
    flat: List[str] = []
    for _, items in groups.items():
        flat.extend(items)
    return flat
