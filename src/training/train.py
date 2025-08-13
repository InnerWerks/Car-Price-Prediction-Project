from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Callable

import itertools
import random
import copy
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
    explained_variance_score,
)
from sklearn.pipeline import Pipeline

from src.data.load import load_raw_df
from src.data.split import split_dataframe
from src.features.tabular import select_features_and_target
from src.models.tabular import build_tabular_model


HIGHER_IS_BETTER = {"r2", "adj_r2", "explained_variance"}
PRIMARY_METRIC = "r2"


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


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    adj_r2 = float(1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - n_features - 1)) if len(y_true) > n_features + 1 else float("nan")
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    smape = float(np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))))
    medae = float(median_absolute_error(y_true, y_pred))
    explained_variance = float(explained_variance_score(y_true, y_pred))
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "adj_r2": adj_r2,
        "mape": mape,
        "smape": smape,
        "medae": medae,
        "explained_variance": explained_variance,
    }


def _save_version_if_best(
    model_path: Path,
    metric_name: str,
    metric_val: float,
    metrics_dir: Path,
    higher_is_better: bool,
) -> bool:
    version_file = metrics_dir.parent / "version.txt"
    best_val: Optional[float] = None
    best_metric: Optional[str] = None
    if version_file.exists():
        content = version_file.read_text().strip().splitlines()
        kv = dict(line.split("=", 1) for line in content if "=" in line)
        best_metric = kv.get("metric")
        if kv.get("value") is not None:
            try:
                best_val = float(kv.get("value"))
            except ValueError:
                best_val = None
    update = False
    if best_metric != metric_name or best_val is None:
        update = True
    else:
        if higher_is_better:
            update = metric_val > best_val
        else:
            update = metric_val < best_val
    if update:
        version_file.write_text(
            f"model_path={model_path}\nmetric={metric_name}\nvalue={metric_val}\n"
        )
        return True
    return False


def _train_single(
    cfg,
    params,
    X_tr,
    y_tr,
    X_va,
    y_va,
    X_te,
    y_te,
    primary_metric: str,
    higher_is_better: bool,
) -> Dict:
    cfg_run = copy.deepcopy(cfg)
    cfg_run.setdefault("model", {}).setdefault("tabular", {})["params"] = params
    model = build_tabular_model(cfg_run, X_tr)
    algo = _algo_name(cfg_run)

    use_es = cfg_run.get("train", {}).get("early_stopping", False)
    needs_manual_es = use_es and algo in {"xgb", "xgboost", "lgbm", "lightgbm"}

    if needs_manual_es:
        pre = model.named_steps["preprocess"]
        reg = model.named_steps["model"]
        pre.fit(X_tr, y_tr)
        X_tr_p = pre.transform(X_tr)
        X_va_p = pre.transform(X_va)
        fit_params, model_params = _fit_params_for_early_stopping(cfg_run, X_va_p, y_va)
        if model_params:
            reg.set_params(**model_params)
        reg.fit(X_tr_p, y_tr, **fit_params)
        model = Pipeline([("preprocess", pre), ("model", reg)])
    else:
        fit_params, model_params = _fit_params_for_early_stopping(cfg_run, X_va, y_va)
        if model_params:
            model.set_params(**{f"model__{k}": v for k, v in model_params.items()})
        fit_params = {f"model__{k}": v for k, v in fit_params.items()}
        model.fit(X_tr, y_tr, **fit_params)

    y_pred_val = model.predict(X_va)
    y_pred_test = model.predict(X_te)
    m_val = _metrics(y_va.values, y_pred_val, X_va.shape[1])
    m_test = _metrics(y_te.values, y_pred_test, X_te.shape[1])

    paths = _paths(cfg_run)
    run_name = _gen_run_name(cfg_run)
    model_path = paths["artifacts"] / f"{run_name}.joblib"
    joblib.dump(model, model_path)

    metrics = {"run": run_name, "algo": _algo_name(cfg_run), "val": m_val, "test": m_test}
    metrics_path = paths["metrics"] / f"{run_name}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    is_best = _save_version_if_best(
        model_path,
        primary_metric,
        m_val[primary_metric],
        paths["metrics"],
        higher_is_better,
    )

    return {
        "run_name": run_name,
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "val_metrics": m_val,
        "test_metrics": m_test,
        "best_updated": is_best,
        "primary_metric": primary_metric,
    }


def train_and_save(cfg, progress_cb: Optional[Callable[[float], None]] = None) -> Dict:
    df, _ = load_raw_df(cfg)
    df_tr, df_va, df_te = split_dataframe(df, cfg)
    X_tr, y_tr = select_features_and_target(df_tr, cfg)
    X_va, y_va = select_features_and_target(df_va, cfg)
    X_te, y_te = select_features_and_target(df_te, cfg)

    search_cfg = cfg.get("model", {}).get("tabular", {}).get("search")
    primary_metric = PRIMARY_METRIC
    higher_is_better = True
    if progress_cb:
        progress_cb(0.0)
    if not search_cfg:
        params = cfg.get("model", {}).get("tabular", {}).get("params", {})
        res = _train_single(
            cfg,
            params,
            X_tr,
            y_tr,
            X_va,
            y_va,
            X_te,
            y_te,
            primary_metric,
            higher_is_better,
        )
        if progress_cb:
            progress_cb(1.0)
        return res

    strategy = search_cfg.get("strategy", "random")
    best: Dict | None = None

    if strategy == "random":
        ranges = search_cfg.get("params", {})
        n_runs = int(search_cfg.get("n_runs", 10))
        for i in range(n_runs):
            params = {}
            for k, (lo, hi) in ranges.items():
                if k in {"n_estimators", "max_depth"}:
                    params[k] = random.randint(int(lo), int(hi))
                else:
                    params[k] = random.uniform(float(lo), float(hi))
            res = _train_single(
                cfg,
                params,
                X_tr,
                y_tr,
                X_va,
                y_va,
                X_te,
                y_te,
                primary_metric,
                higher_is_better,
            )
            if not best:
                best = res
            else:
                cur = res["val_metrics"][primary_metric]
                best_val = best["val_metrics"][primary_metric]
                if (cur > best_val and higher_is_better) or (cur < best_val and not higher_is_better):
                    best = res
            if progress_cb:
                progress_cb((i + 1) / n_runs)
    elif strategy == "grid":
        grid = []
        for k, rng in search_cfg.get("params", {}).items():
            if k in {"n_estimators", "max_depth"}:
                vals = list(range(int(rng["min"]), int(rng["max"]) + 1, int(rng["step"])))
            else:
                vals = list(
                    np.arange(
                        float(rng["min"]),
                        float(rng["max"]) + float(rng["step"]) / 2,
                        float(rng["step"]),
                    )
                )
            grid.append((k, vals))
        combos = list(itertools.product(*[v for _, v in grid]))
        total = len(combos)
        for i, combo in enumerate(combos):
            params = {grid[j][0]: combo[j] for j in range(len(grid))}
            res = _train_single(
                cfg,
                params,
                X_tr,
                y_tr,
                X_va,
                y_va,
                X_te,
                y_te,
                primary_metric,
                higher_is_better,
            )
            if not best:
                best = res
            else:
                cur = res["val_metrics"][primary_metric]
                best_val = best["val_metrics"][primary_metric]
                if (cur > best_val and higher_is_better) or (cur < best_val and not higher_is_better):
                    best = res
            if progress_cb:
                progress_cb((i + 1) / total)
    else:  # fallback
        params = cfg.get("model", {}).get("tabular", {}).get("params", {})
        res = _train_single(
            cfg,
            params,
            X_tr,
            y_tr,
            X_va,
            y_va,
            X_te,
            y_te,
            primary_metric,
            higher_is_better,
        )
        if progress_cb:
            progress_cb(1.0)
        best = res

    assert best is not None
    return best

