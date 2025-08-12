from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


def _processed_dir(cfg) -> Path:
    return Path(cfg["dataset"]["paths"]["processed"]).resolve()


def _artifacts_dir(cfg) -> Path:
    return Path("models/artifacts").resolve()


def _feature_config(cfg) -> Dict:
    ds = cfg.get("dataset", {})
    feats = ds.get("features", {})
    include = feats.get("include", []) or []
    drop = feats.get("drop", []) or []
    target = ds.get("target")
    if not target:
        raise ValueError("Config missing dataset.target")
    return {"include": include, "drop": drop, "target": target}


def _select_columns(df: pd.DataFrame, feature_cfg: Dict) -> Tuple[pd.DataFrame, pd.Series]:
    target = feature_cfg["target"]
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in data")
    drops = set(feature_cfg["drop"]) | {target}
    if feature_cfg["include"]:
        X = df[feature_cfg["include"]].copy()
    else:
        X = df.drop(columns=[c for c in drops if c in df.columns], errors="ignore").copy()
    y = df[target].copy()
    return X, y


def _build_preprocessor(cfg, X: pd.DataFrame) -> ColumnTransformer:
    tr = cfg.get("train", {}).get("preprocess", cfg.get("preprocess", {}))
    impute_num = (tr.get("impute", {}) or {}).get("numeric", "median")
    impute_cat = (tr.get("impute", {}) or {}).get("categorical", "most_frequent")
    encode_cat = tr.get("encode", {}).get("categorical", "onehot")
    scale = tr.get("scale_numeric", "standard")

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    num_steps = [("imputer", SimpleImputer(strategy=impute_num))]
    if scale == "standard":
        num_steps.append(("scaler", StandardScaler()))
    elif scale == "minmax":
        num_steps.append(("scaler", MinMaxScaler()))

    if encode_cat == "onehot":
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            # for older sklearn
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    else:
        # fallback: one-hot if unsupported option provided
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_steps = [("imputer", SimpleImputer(strategy=impute_cat)), ("encoder", ohe)]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), num_cols),
            ("cat", Pipeline(cat_steps), cat_cols),
        ],
        remainder="drop",
    )
    return pre


def _transform_to_frame(pre: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    Xt = pre.transform(X)
    # Column names
    try:
        cols = pre.get_feature_names_out()
    except Exception:
        cols = [f"f{i}" for i in range(Xt.shape[1])]
    return pd.DataFrame(Xt, columns=cols)


def preprocess_and_persist(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, cfg) -> None:
    feature_cfg = _feature_config(cfg)
    X_tr, y_tr = _select_columns(df_train, feature_cfg)
    X_va, y_va = _select_columns(df_val, feature_cfg)
    X_te, y_te = _select_columns(df_test, feature_cfg)

    pre = _build_preprocessor(cfg, X_tr)
    Xtr = pre.fit_transform(X_tr)
    Xva = pre.transform(X_va)
    Xte = pre.transform(X_te)

    # Convert to DataFrame with names
    try:
        cols = pre.get_feature_names_out()
    except Exception:
        cols = [f"f{i}" for i in range(Xtr.shape[1])]
    df_Xtr = pd.DataFrame(Xtr, columns=cols)
    df_Xva = pd.DataFrame(Xva, columns=cols)
    df_Xte = pd.DataFrame(Xte, columns=cols)

    # Re-attach target
    df_tr_proc = pd.concat([df_Xtr, y_tr.reset_index(drop=True)], axis=1)
    df_va_proc = pd.concat([df_Xva, y_va.reset_index(drop=True)], axis=1)
    df_te_proc = pd.concat([df_Xte, y_te.reset_index(drop=True)], axis=1)

    # Persist
    pdir = _processed_dir(cfg)
    pdir.mkdir(parents=True, exist_ok=True)
    df_tr_proc.to_csv(pdir / "train.csv", index=False)
    df_va_proc.to_csv(pdir / "val.csv", index=False)
    df_te_proc.to_csv(pdir / "test.csv", index=False)

    # Save preprocessor
    adir = _artifacts_dir(cfg)
    adir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, adir / "preprocessor.joblib")

