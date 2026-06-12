from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


OD_COLUMNS = {"start_station", "end_line", "passenger_count", "time_window"}
STATIC_FILES = (
    "station_features.xlsx",
    "line_features.xlsx",
    "bus_departure_info_16-20.xlsx",
)


@dataclass(frozen=True)
class PreparedDataSummary:
    copied_static_files: list[Path]
    copied_od_files: list[Path]


def copy_existing_dataset(
    source_root: Path,
    project_root: Path,
    date: str = "20190501",
) -> PreparedDataSummary:
    static_dir = project_root / "data" / "static"
    od_dir = project_root / "data" / "processed" / "od_by_time"
    static_dir.mkdir(parents=True, exist_ok=True)
    od_dir.mkdir(parents=True, exist_ok=True)

    copied_static: list[Path] = []
    for filename in STATIC_FILES:
        src = source_root / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing required static file: {src}")
        dst = static_dir / filename
        shutil.copy2(src, dst)
        copied_static.append(dst)

    copied_od: list[Path] = []
    od_files = sorted(source_root.glob(f"{date}_*.csv"))
    if not od_files:
        raise FileNotFoundError(f"No OD CSV files matched {source_root / (date + '_*.csv')}")

    for src in od_files:
        validate_od_file(src)
        dst = od_dir / src.name
        shutil.copy2(src, dst)
        copied_od.append(dst)

    return PreparedDataSummary(copied_static, copied_od)


def validate_od_file(path: Path) -> None:
    sample = pd.read_csv(path, nrows=5)
    missing = OD_COLUMNS.difference(sample.columns)
    if missing:
        raise ValueError(f"{path} is missing OD columns: {sorted(missing)}")


def discover_od_files(od_dir: Path, date: str) -> list[Path]:
    return sorted(od_dir.glob(f"{date}_*.csv"))


def validate_prepared_dataset(static_dir: Path, od_dir: Path, date: str) -> None:
    for filename in STATIC_FILES:
        path = static_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required static file: {path}")

    files = discover_od_files(od_dir, date)
    if not files:
        raise FileNotFoundError(f"No prepared OD CSV files found in {od_dir}")
    for path in files:
        validate_od_file(path)


def load_station_features(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    required = {"start_station", "end_line", "end_station", "time", "line"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing station feature columns: {sorted(missing)}")
    return df


def load_line_features(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    required = {"busy_line", "free_line", "dispatch_to"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing line feature columns: {sorted(missing)}")
    return df


def load_departure_info(path: Path) -> np.ndarray:
    df = pd.read_excel(path)
    required = {"Bus_Line", "Departure_Time"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing departure columns: {sorted(missing)}")
    return df.to_numpy(dtype=object)


def od_file_for_time(od_dir: Path, date: str, trigger_time) -> Path:
    time_str = trigger_time.strftime("%H%M%S")
    return od_dir / f"{date}_{time_str}.csv"


def allocate_od_to_station_edges(od_df: pd.DataFrame, station_features_df: pd.DataFrame) -> np.ndarray:
    features = station_features_df.loc[:, ["start_station", "end_line", "end_station", "time", "line"]].copy()
    od = od_df.loc[:, ["start_station", "end_line", "passenger_count"]].copy()
    merged = pd.merge(features, od, on=["start_station", "end_line"], how="left")
    merged["allocated_passengers"] = 0

    for (_start, _line), group in merged.groupby(["start_station", "end_line"]):
        total = group["passenger_count"].iloc[0]
        if pd.isna(total) or total <= 0:
            continue
        exp_time = np.exp(-group["time"].astype(float))
        denom = exp_time.sum()
        if denom <= 0:
            continue
        allocated = (exp_time / denom * float(total)).round().astype(int)
        merged.loc[group.index, "allocated_passengers"] = allocated

    return merged["allocated_passengers"].to_numpy(dtype=int)
