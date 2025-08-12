from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.features.tabular import build_preprocessor


def _get_model_cfg(cfg) -> Tuple[str, Dict[str, Any]]:
    mcfg = cfg.get("model", {}).get("tabular", {})
    algo = mcfg.get("algorithm", "linear").lower()
    params = mcfg.get("params", {}) or {}
    return algo, params


def _make_regressor(algorithm: str, params: Dict[str, Any]):
    algorithm = algorithm.lower()
    if algorithm in {"linear", "linreg", "ols"}:
        from sklearn.linear_model import LinearRegression

        return LinearRegression(**params)
    elif algorithm in {"ridge"}:
        from sklearn.linear_model import Ridge

        return Ridge(**params)
    elif algorithm in {"lasso"}:
        from sklearn.linear_model import Lasso

        return Lasso(**params)
    elif algorithm in {"rf", "randomforest", "random_forest"}:
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(**params)
    elif algorithm in {"xgb", "xgboost"}:
        try:
            from xgboost import XGBRegressor
        except Exception as e:
            raise ImportError("xgboost is not installed. Install it or switch algorithm.") from e

        return XGBRegressor(**params)
    elif algorithm in {"lgbm", "lightgbm"}:
        try:
            from lightgbm import LGBMRegressor
        except Exception as e:
            raise ImportError("lightgbm is not installed. Install it or switch algorithm.") from e

        return LGBMRegressor(**params)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def build_tabular_model(cfg, X_sample: pd.DataFrame) -> Pipeline:
    """
    Build a modeling pipeline for tabular regression:
    - Preprocessor (impute/encode/scale) per config
    - Regressor per config with params
    """
    algo, params = _get_model_cfg(cfg)
    pre = build_preprocessor(cfg, X_sample)
    reg = _make_regressor(algo, params)
    pipe = Pipeline([
        ("preprocess", pre),
        ("model", reg),
    ])
    return pipe

