"""
Brain concept map: load brain.yaml and expose flattened nodes for visualization.
Used by Brain Concept Visualization (2.1.0) in conversation UI.
"""

from pathlib import Path
from typing import Any, Dict, List

import yaml

_BRAIN_YAML = Path(__file__).resolve().parent.parent / "brain.yaml"


def load_brain_nodes() -> List[Dict[str, Any]]:
    """
    Load brain.yaml and return a flat list of nodes.
    Each node: { "id": str, "label": str, "lobe": str, "x": float, "y": float }.
    """
    if not _BRAIN_YAML.exists():
        return []
    try:
        with open(_BRAIN_YAML, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (yaml.YAMLError, IOError):
        return []
    if not isinstance(data, dict):
        return []
    nodes: List[Dict[str, Any]] = []
    for lobe, concepts in data.items():
        if not isinstance(concepts, dict):
            continue
        for concept_name, coords in concepts.items():
            if not isinstance(coords, (list, tuple)) or len(coords) < 2:
                continue
            try:
                x, y = float(coords[0]), float(coords[1])
            except (TypeError, ValueError):
                continue
            node_id = f"{lobe}/{concept_name}"
            nodes.append({
                "id": node_id,
                "label": concept_name.replace("_", " "),
                "lobe": lobe,
                "x": x,
                "y": y,
            })
    return nodes
