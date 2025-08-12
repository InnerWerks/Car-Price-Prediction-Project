from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def _interim_dir(cfg) -> Path:
    return Path(cfg["dataset"]["paths"]["interim"]).resolve()


def split_dataframe(df: pd.DataFrame, cfg) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_cfg = cfg["dataset"].get("split", {})
    test_size = float(split_cfg.get("test_size", 0.2))
    val_size = float(split_cfg.get("val_size", 0.1))
    seed = int(cfg["dataset"].get("seed", cfg.get("seed", 42)))

    # First take test split
    df_trainval, df_test = train_test_split(df, test_size=test_size, random_state=seed)
    # Then split train/val
    val_rel = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(df_trainval, test_size=val_rel, random_state=seed)
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


def persist_interim_splits(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, cfg) -> None:
    idir = _interim_dir(cfg)
    idir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(idir / "train.csv", index=False)
    df_val.to_csv(idir / "val.csv", index=False)
    df_test.to_csv(idir / "test.csv", index=False)

