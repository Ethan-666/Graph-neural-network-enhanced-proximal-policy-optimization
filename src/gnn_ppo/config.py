from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return project_root() / "configs" / "default.json"


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else default_config_path()
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_root() / path


def data_paths(config: dict[str, Any]) -> dict[str, Path]:
    data_cfg = config["data"]
    static_dir = resolve_project_path(data_cfg["static_dir"])
    od_dir = resolve_project_path(data_cfg["od_dir"])
    output_dir = resolve_project_path(data_cfg["output_dir"])
    return {
        "static_dir": static_dir,
        "od_dir": od_dir,
        "output_dir": output_dir,
        "station_features": static_dir / data_cfg["station_features"],
        "line_features": static_dir / data_cfg["line_features"],
        "departure_info": static_dir / data_cfg["departure_info"],
    }

