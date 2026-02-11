from pathlib import Path
from typing import Any, Dict
import yaml
from flask import Blueprint, jsonify, request


theme_bp = Blueprint("theme", __name__)

_CUSTOM_CONFIG_PATH = Path(__file__).resolve().parents[2] / "custom_config.yaml"


def _load_custom_config() -> Dict[str, Any]:
    if not _CUSTOM_CONFIG_PATH.exists():
        return {}
    try:
        with open(_CUSTOM_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _save_custom_config(data: Dict[str, Any]) -> None:
    with open(_CUSTOM_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def _get_overrides() -> Dict[str, Dict[str, str]]:
    data = _load_custom_config()
    raw = data.get("THEME_OVERRIDES") if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        return {"light": {}, "dark": {}}
    light = raw.get("light") if isinstance(raw.get("light"), dict) else {}
    dark = raw.get("dark") if isinstance(raw.get("dark"), dict) else {}
    return {"light": dict(light), "dark": dict(dark)}


def _write_overrides(overrides: Dict[str, Dict[str, str]]) -> None:
    data = _load_custom_config()
    if not isinstance(data, dict):
        data = {}
    data["THEME_OVERRIDES"] = overrides
    _save_custom_config(data)


@theme_bp.get("/api/theme/custom")
def api_get_custom_theme():
    return jsonify({"overrides": _get_overrides()})


@theme_bp.post("/api/theme/custom")
def api_set_custom_theme():
    payload = request.get_json(force=True) or {}
    overrides = _get_overrides()
    if "overrides" in payload and isinstance(payload.get("overrides"), dict):
        raw = payload["overrides"]
        for theme in ("light", "dark"):
            if isinstance(raw.get(theme), dict):
                overrides[theme] = dict(raw[theme])
    else:
        theme = (payload.get("theme") or "").strip().lower()
        key = (payload.get("key") or "").strip()
        value = (payload.get("value") or "").strip()
        if theme not in ("light", "dark") or not key.startswith("--"):
            return jsonify({"error": "invalid payload"}), 400
        overrides[theme] = overrides.get(theme) or {}
        if value:
            overrides[theme][key] = value
        else:
            overrides[theme].pop(key, None)
    _write_overrides(overrides)
    return jsonify({"status": "ok", "overrides": overrides})
