from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def exponential_moving_average(values: list[float] | np.ndarray, alpha: float = 0.1) -> np.ndarray:
    if len(values) == 0:
        return np.array([], dtype=float)
    smoothed = [float(values[0])]
    for value in values[1:]:
        smoothed.append(alpha * float(value) + (1.0 - alpha) * smoothed[-1])
    return np.asarray(smoothed, dtype=float)


def timestamp_for_run(seed: int) -> str:
    from datetime import datetime

    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{seed}"

