from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def _raw_dir(cfg) -> Path:
    return Path(cfg["dataset"]["paths"]["raw"]).resolve()


def _choose_raw_csv(raw_dir: Path, filename: Optional[str] = None) -> Path:
    if filename:
        p = raw_dir / filename
        if not p.exists():
            raise FileNotFoundError(f"Raw file not found: {p}")
        return p
    csvs = sorted(raw_dir.glob("*.csv"))
    if len(csvs) == 0:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}. Place a CSV there or specify filename.")
    if len(csvs) > 1:
        names = "\n - ".join(str(c.name) for c in csvs)
        raise RuntimeError(
            "Multiple CSV files found in data/raw. Specify which to use via filename or config.\nFound:\n - "
            + names
        )
    return csvs[0]


def load_raw_df(cfg, filename: Optional[str] = None) -> Tuple[pd.DataFrame, Path]:
    """
    Load the raw dataset from data/raw as a DataFrame.

    - If `filename` is provided, loads that file.
    - Else, loads the single CSV in data/raw; errors if none or multiple.
    """
    raw_dir = _raw_dir(cfg)
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = _choose_raw_csv(raw_dir, filename)
    df = pd.read_csv(csv_path)
    return df, csv_path

