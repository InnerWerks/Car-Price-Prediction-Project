from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.data.load import load_raw_df
from src.data.split import split_dataframe
from src.features.tabular import select_features_and_target
from src.models.tabular import build_tabular_model


def _paths(cfg) -> Dict[str, Path]:
    artifacts_dir = Path(cfg.get("train", {}).get("logging", {}).get("artifacts_dir", "models/artifacts")).resolve()
    metrics_dir = Path(cfg.get("train", {}).get("logging", {}).get("metrics_dir", "models/metrics")).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return {"artifacts": artifacts_dir, "metrics": metrics_dir}


def _gen_run_name(cfg) -> str:
    base = cfg.get("train", {}).get("logging", {}).get("run_name", "run")
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"{base}_{ts}"


def _algo_name(cfg) -> str:
    return cfg.get("model", {}).get("tabular", {}).get("algorithm", "").lower()


def _fit_params_for_early_stopping(cfg, X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[Dict, Dict]:
    """Generate fit and model params for early stopping.

    XGBoost 2.1 removed ``early_stopping_rounds`` from ``fit`` in the
    scikit-learn API, so this helper now returns two dictionaries:

    - ``fit_params`` – parameters passed to the estimator's ``fit``
    - ``model_params`` – parameters set on the estimator prior to ``fit``
    """

    early = cfg.get("train", {}).get("early_stopping", False)
    rounds = int(cfg.get("train", {}).get("early_stopping_rounds", 50))
    if not early:
        return {}, {}

    algo = _algo_name(cfg)
    if algo in {"xgb", "xgboost"}:
        # ``early_stopping_rounds`` must be configured on the estimator
        fit_params = {"eval_set": [(X_val, y_val)], "verbose": False}
        model_params = {"early_stopping_rounds": rounds}
        return fit_params, model_params
    elif algo in {"lgbm", "lightgbm"}:
        # LightGBM sklearn API supports callbacks for early stopping
        try:
            import lightgbm as lgb

            fit_params = {
                "eval_set": [(X_val, y_val)],
                "callbacks": [lgb.early_stopping(rounds, verbose=False)],
            }
            return fit_params, {}
        except Exception:
            return {"eval_set": [(X_val, y_val)]}, {}

    return {}, {}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _save_version_if_best(model_path: Path, val_rmse: float, metrics_dir: Path) -> bool:
    version_file = model_path.parent.parent / "version.txt"
    best_rmse: Optional[float] = None
    if version_file.exists():
        content = version_file.read_text().strip().splitlines()
        kv = dict(line.split("=", 1) for line in content if "=" in line)
        if "val_rmse" in kv:
            try:
                best_rmse = float(kv["val_rmse"])
            except ValueError:
                best_rmse = None
    if best_rmse is None or val_rmse < best_rmse:
        version_file.write_text(f"model_path={model_path}\nval_rmse={val_rmse}\n")
        return True
    return False


def train_and_save(cfg) -> Dict:
    # Load raw data and split
    df, _ = load_raw_df(cfg)
    df_tr, df_va, df_te = split_dataframe(df, cfg)
    X_tr, y_tr = select_features_and_target(df_tr, cfg)
    X_va, y_va = select_features_and_target(df_va, cfg)
    X_te, y_te = select_features_and_target(df_te, cfg)

    # Build model pipeline
    model = build_tabular_model(cfg, X_tr)
    algo = _algo_name(cfg)

    use_es = cfg.get("train", {}).get("early_stopping", False)
    needs_manual_es = use_es and algo in {"xgb", "xgboost", "lgbm", "lightgbm"}

    if needs_manual_es:
        # Preprocess training/validation before passing to estimator.fit so
        # evaluation matrices don't contain raw categorical dtypes.
        pre = model.named_steps["preprocess"]
        reg = model.named_steps["model"]
        pre.fit(X_tr, y_tr)
        X_tr_p = pre.transform(X_tr)
        X_va_p = pre.transform(X_va)
        fit_params, model_params = _fit_params_for_early_stopping(cfg, X_va_p, y_va)
        if model_params:
            reg.set_params(**model_params)
        reg.fit(X_tr_p, y_tr, **fit_params)
        model = Pipeline([("preprocess", pre), ("model", reg)])
    else:
        fit_params, model_params = _fit_params_for_early_stopping(cfg, X_va, y_va)
        if model_params:
            model.set_params(**{f"model__{k}": v for k, v in model_params.items()})
        fit_params = {f"model__{k}": v for k, v in fit_params.items()}
        model.fit(X_tr, y_tr, **fit_params)

    # Evaluate on val and test
    y_pred_val = model.predict(X_va)
    y_pred_test = model.predict(X_te)
    m_val = _metrics(y_va.values, y_pred_val)
    m_test = _metrics(y_te.values, y_pred_test)

    # Save model and metrics
    paths = _paths(cfg)
    run_name = _gen_run_name(cfg)
    model_path = paths["artifacts"] / f"{run_name}.joblib"
    joblib.dump(model, model_path)

    metrics = {"run": run_name, "algo": _algo_name(cfg), "val": m_val, "test": m_test}
    metrics_path = paths["metrics"] / f"{run_name}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    is_best = _save_version_if_best(model_path, m_val["rmse"], paths["metrics"])

    return {
        "run_name": run_name,
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "val_metrics": m_val,
        "test_metrics": m_test,
        "best_updated": is_best,
    }

