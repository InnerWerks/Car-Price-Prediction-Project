from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
    explained_variance_score,
)
from sklearn.model_selection import learning_curve
from sklearn.inspection import PartialDependenceDisplay
from sklearn.base import clone

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


def _figures_dir() -> Path:
    d = Path("reports/figures")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _plot_and_save(model, X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, cfg, run_tag: str) -> Dict[str, str]:
    figs: Dict[str, str] = {}
    figdir = _figures_dir()
    plots_cfg = cfg.get("eval", {}).get("plots", {})
    residuals = y_true - y_pred

    if plots_cfg.get("pred_vs_actual"):
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        plt.plot(lims, lims, 'r--', linewidth=1)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Predicted vs Actual')
        p = figdir / f"pred_vs_actual_{run_tag}.png"
        plt.tight_layout(); plt.savefig(p); plt.close()
        figs['pred_vs_actual'] = str(p)

    if plots_cfg.get("prediction_error_hist"):
        plt.figure(figsize=(6,4))
        plt.hist(y_pred - y_true, bins=30, alpha=0.7)
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        plt.title('Prediction Error Histogram')
        p = figdir / f"prediction_error_hist_{run_tag}.png"
        plt.tight_layout(); plt.savefig(p); plt.close()
        figs['prediction_error_hist'] = str(p)

    if plots_cfg.get("residuals_vs_fitted"):
        plt.figure(figsize=(6,4))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(0, color='r', linestyle='--', linewidth=1)
        plt.xlabel('Fitted')
        plt.ylabel('Residual')
        plt.title('Residuals vs Fitted')
        p = figdir / f"residuals_vs_fitted_{run_tag}.png"
        plt.tight_layout(); plt.savefig(p); plt.close()
        figs['residuals_vs_fitted'] = str(p)

    if plots_cfg.get("residual_hist"):
        plt.figure(figsize=(6,4))
        plt.hist(residuals, bins=30, alpha=0.7)
        plt.xlabel('Residual')
        plt.ylabel('Count')
        plt.title('Residual Histogram')
        p = figdir / f"residual_hist_{run_tag}.png"
        plt.tight_layout(); plt.savefig(p); plt.close()
        figs['residual_hist'] = str(p)

    if plots_cfg.get("residual_qq"):
        plt.figure(figsize=(6,6))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Residual Q-Q')
        p = figdir / f"residual_qq_{run_tag}.png"
        plt.tight_layout(); plt.savefig(p); plt.close()
        figs['residual_qq'] = str(p)

    if plots_cfg.get("learning_curve"):
        model_lc = clone(model)
        reg = model_lc.named_steps.get("model", model_lc)
        if hasattr(reg, "early_stopping_rounds"):
            reg.set_params(early_stopping_rounds=0)
        train_sizes, train_scores, val_scores = learning_curve(
            model_lc, X_tr, y_tr, cv=5, scoring="neg_root_mean_squared_error"
        )
        plt.figure(figsize=(6,4))
        plt.plot(train_sizes, -train_scores.mean(axis=1), label='Train')
        plt.plot(train_sizes, -val_scores.mean(axis=1), label='CV')
        plt.xlabel('Training examples')
        plt.ylabel('RMSE')
        plt.title('Learning Curve')
        plt.legend()
        p = figdir / f"learning_curve_{run_tag}.png"
        plt.tight_layout(); plt.savefig(p); plt.close()
        figs['learning_curve'] = str(p)

    if plots_cfg.get("feature_importances"):
        reg = model.named_steps.get("model", model)
        if hasattr(reg, "feature_importances_"):
            pre = model.named_steps.get("preprocess")
            try:
                names = pre.get_feature_names_out()
            except Exception:
                names = [f"f{i}" for i in range(len(reg.feature_importances_))]
            order = np.argsort(reg.feature_importances_)[::-1][:20]
            plt.figure(figsize=(6,4))
            plt.barh(range(len(order)), reg.feature_importances_[order][::-1])
            plt.yticks(range(len(order)), [names[i] for i in order][::-1])
            plt.title('Feature Importances')
            p = figdir / f"feature_importances_{run_tag}.png"
            plt.tight_layout(); plt.savefig(p); plt.close()
            figs['feature_importances'] = str(p)

    if plots_cfg.get("pdp_1d") and X_tr.shape[1] >= 1:
        feature = X_tr.columns[0]
        fig = PartialDependenceDisplay.from_estimator(model, X_tr, [feature])
        p = figdir / f"pdp_1d_{run_tag}.png"
        fig.figure_.tight_layout(); fig.figure_.savefig(p); plt.close(fig.figure_)
        figs['pdp_1d'] = str(p)

    if plots_cfg.get("pdp_2d") and X_tr.shape[1] >= 2:
        feats = [X_tr.columns[0], X_tr.columns[1]]
        fig = PartialDependenceDisplay.from_estimator(model, X_tr, [feats])
        p = figdir / f"pdp_2d_{run_tag}.png"
        fig.figure_.tight_layout(); fig.figure_.savefig(p); plt.close(fig.figure_)
        figs['pdp_2d'] = str(p)

    return figs


def evaluate_saved_model(cfg, model_path: Optional[str] = None) -> Dict:
    # Resolve model path
    model_file = Path(model_path) if model_path else _best_model_path()
    if not model_file or not model_file.exists():
        raise FileNotFoundError("No model found. Provide --model-path or ensure models/version.txt exists.")

    # Load data and create splits
    df, _ = load_raw_df(cfg)
    df_tr, _, df_te = split_dataframe(df, cfg)
    X_tr, y_tr = select_features_and_target(df_tr, cfg)
    X_te, y_te = select_features_and_target(df_te, cfg)

    model = joblib.load(model_file)
    y_pred = model.predict(X_te)
    m = _metrics(y_te.values, y_pred, X_te.shape[1])

    tag = model_file.stem
    figs = _plot_and_save(model, X_tr, y_tr, X_te, y_te.values, y_pred, cfg, run_tag=tag)

    # Save metrics JSON alongside
    metrics_dir = Path("models/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out = {"model_path": str(model_file), "test": m, "figures": figs}
    (metrics_dir / f"eval_{tag}.json").write_text(json.dumps(out, indent=2))
    return out

