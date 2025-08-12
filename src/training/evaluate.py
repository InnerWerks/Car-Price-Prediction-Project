from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.load import load_raw_df
from src.data.split import split_dataframe
from src.features.tabular import select_features_and_target


def _best_model_path() -> Optional[Path]:
    version_file = Path("models/version.txt")
    if not version_file.exists():
        return None
    kv = {}
    for line in version_file.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    p = kv.get("model_path")
    return Path(p) if p else None


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _figures_dir() -> Path:
    d = Path("reports/figures")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _plot_and_save(y_true: np.ndarray, y_pred: np.ndarray, run_tag: str) -> Dict[str, str]:
    figs = {}
    figdir = _figures_dir()

    # Predicted vs Actual
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', linewidth=1)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Actual')
    p1 = figdir / f"pred_vs_actual_{run_tag}.png"
    plt.tight_layout(); plt.savefig(p1); plt.close()
    figs['pred_vs_actual'] = str(p1)

    # Residuals
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.xlabel('Residual (y - y_pred)')
    plt.ylabel('Count')
    plt.title('Residuals Distribution')
    p2 = figdir / f"residuals_{run_tag}.png"
    plt.tight_layout(); plt.savefig(p2); plt.close()
    figs['residuals'] = str(p2)

    return figs


def evaluate_saved_model(cfg, model_path: Optional[str] = None) -> Dict:
    # Resolve model path
    model_file = Path(model_path) if model_path else _best_model_path()
    if not model_file or not model_file.exists():
        raise FileNotFoundError("No model found. Provide --model-path or ensure models/version.txt exists.")

    # Load data (raw) and create test set
    df, _ = load_raw_df(cfg)
    _, _, df_te = split_dataframe(df, cfg)
    X_te, y_te = select_features_and_target(df_te, cfg)

    # Load model and predict
    model = joblib.load(model_file)
    y_pred = model.predict(X_te)
    m = _metrics(y_te.values, y_pred)

    # Save figures
    tag = model_file.stem
    figs = _plot_and_save(y_te.values, y_pred, run_tag=tag)

    # Save metrics JSON alongside
    metrics_dir = Path("models/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out = {"model_path": str(model_file), "test": m, "figures": figs}
    (metrics_dir / f"eval_{tag}.json").write_text(json.dumps(out, indent=2))
    return out

