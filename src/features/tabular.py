from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin


class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    """
    Simple target mean encoder for categorical features.

    Note: This implementation is minimal and does not include
    cross-fold target leakage prevention. Use for baselines only.
    """

    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns
        self.mapping_: Dict[str, Dict[object, float]] = {}
        self.global_mean_: float = np.nan

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.columns is None:
            self.columns = [c for c in X.columns]
        self.global_mean_ = float(np.mean(y))
        self.mapping_ = {}
        for c in self.columns:
            m = (
                pd.DataFrame({"x": X[c], "y": y})
                .groupby("x")["y"].mean()
                .to_dict()
            )
            self.mapping_[c] = m
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.columns:
            m = self.mapping_.get(c, {})
            X[c] = X[c].map(m).fillna(self.global_mean_)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns if input_features is None else input_features)


def _split_numeric_categorical(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    return num_cols, cat_cols


def build_preprocessor(cfg, X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer according to config:
    - Impute numeric/categorical
    - Encode categorical via onehot or target encoding
    - Scale numeric via standard or minmax
    """
    tr = cfg.get("train", {}).get("preprocess", cfg.get("preprocess", {}))
    impute_num = (tr.get("impute", {}) or {}).get("numeric", "median")
    impute_cat = (tr.get("impute", {}) or {}).get("categorical", "most_frequent")
    encode_cat = tr.get("encode", {}).get("categorical", "onehot")
    scale = tr.get("scale_numeric", "standard")

    num_cols, cat_cols = _split_numeric_categorical(X)

    num_steps = [("imputer", SimpleImputer(strategy=impute_num))]
    if scale == "standard":
        num_steps.append(("scaler", StandardScaler()))
    elif scale == "minmax":
        num_steps.append(("scaler", MinMaxScaler()))

    if encode_cat == "onehot":
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    elif encode_cat == "target":
        encoder = TargetMeanEncoder(columns=cat_cols)
    else:
        # default to onehot
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_steps: List[Tuple[str, TransformerMixin]] = [("imputer", SimpleImputer(strategy=impute_cat))]
    # TargetMeanEncoder needs y during fit; handled by sklearn Pipeline fit(X, y)
    cat_steps.append(("encoder", encoder))

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), num_cols),
            ("cat", Pipeline(cat_steps), cat_cols),
        ],
        remainder="drop",
    )
    return pre


def select_features_and_target(df: pd.DataFrame, cfg) -> Tuple[pd.DataFrame, pd.Series]:
    ds = cfg.get("dataset", {})
    feats = ds.get("features", {})
    include = feats.get("include", []) or []
    drop = feats.get("drop", []) or []
    target = ds.get("target")
    if not target:
        raise ValueError("Config missing dataset.target")
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in data")

    drops = set(drop) | {target}
    if include:
        X = df[include].copy()
    else:
        X = df.drop(columns=[c for c in drops if c in df.columns], errors="ignore").copy()
    y = df[target].copy()
    return X, y


# Optional utilities for feature importance based pruning
def select_top_k_by_importance(feature_names: Iterable[str], importances: Iterable[float], top_k: int) -> List[str]:
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return [name for name, _ in pairs[:top_k]]

