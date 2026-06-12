from __future__ import annotations

from typing import Any


def transform_metric(reporting_config: dict[str, Any], metric_name: str, value: float) -> float:
    metric_config = reporting_config.get(metric_name, {})
    transformed = float(value) * float(metric_config.get("scale", 1.0)) + float(metric_config.get("offset", 0.0))

    clip_min = metric_config.get("clip_min")
    clip_max = metric_config.get("clip_max")
    if clip_min is not None:
        transformed = max(transformed, float(clip_min))
    if clip_max is not None:
        transformed = min(transformed, float(clip_max))

    if bool(metric_config.get("round", False)):
        return float(round(transformed))
    return transformed

