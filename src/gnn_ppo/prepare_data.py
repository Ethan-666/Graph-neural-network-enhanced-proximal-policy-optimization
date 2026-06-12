from __future__ import annotations

import argparse
from pathlib import Path

from .config import project_root
from .data import copy_existing_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the reproduction dataset from existing 20190501 OD CSV files and static Excel files."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=project_root().parent,
        help="Directory containing station_features.xlsx, line_features.xlsx, bus_departure_info_16-20.xlsx, and 20190501_*.csv.",
    )
    parser.add_argument("--date", default="20190501", help="OD CSV date prefix to copy.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = copy_existing_dataset(args.source_root.resolve(), project_root(), date=args.date)
    print("Prepared dataset")
    print(f"Static files: {len(summary.copied_static_files)}")
    for path in summary.copied_static_files:
        print(f"  - {path}")
    print(f"OD CSV files: {len(summary.copied_od_files)}")
    print(f"First OD file: {summary.copied_od_files[0]}")
    print(f"Last OD file : {summary.copied_od_files[-1]}")


if __name__ == "__main__":
    main()
