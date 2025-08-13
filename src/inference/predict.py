from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Tuple

import joblib
import pandas as pd

__all__ = ["predict_from_csv", "best_model_path", "best_model_info"]


def best_model_info() -> Optional[Dict[str, object]]:
    """Return info about the best model from ``models/version.txt`` if available."""
    version_file = Path("models/version.txt")
    if not version_file.exists():
        return None
    kv: Dict[str, str] = {}
    for line in version_file.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    model_path = kv.get("model_path")
    metric_name = kv.get("metric")
    metric_val = None
    if kv.get("value") is not None:
        try:
            metric_val = float(kv.get("value"))
        except ValueError:
            metric_val = None
    if not model_path:
        return None
    return {
        "model_path": Path(model_path),
        "metric": metric_name,
        "value": metric_val,
    }


def best_model_path() -> Optional[Path]:
    info = best_model_info()
    return info["model_path"] if info else None


def _select_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Select feature columns from ``df`` based on dataset config."""
    ds = cfg.get("dataset", {})
    feats = ds.get("features", {})
    include = feats.get("include", []) or []
    drop = feats.get("drop", []) or []
    target = ds.get("target")
    if target and target in df.columns:
        df = df.drop(columns=[target])
    if include:
        return df[include].copy()
    return df.drop(columns=[c for c in drop if c in df.columns], errors="ignore").copy()


def _validate_input(df: pd.DataFrame, model) -> None:
    """Validate that ``df`` matches the model's expected feature schema."""
    if df.empty:
        raise ValueError("Input CSV has no rows.")
    if df.shape[1] == 0:
        raise ValueError("No feature columns available for prediction after preprocessing.")
    if df.isnull().any().any():
        cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(
            "Input CSV contains missing values in columns: " + ", ".join(cols)
        )
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        missing = [c for c in expected if c not in df.columns]
        unexpected = [c for c in df.columns if c not in expected]
        if missing or unexpected:
            parts = []
            if missing:
                parts.append("missing columns: " + ", ".join(missing))
            if unexpected:
                parts.append("unexpected columns: " + ", ".join(unexpected))
            raise ValueError("Input CSV columns mismatch – " + "; ".join(parts))


def predict_from_csv(
    cfg,
    csv_path: str,
    model_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, str]:
    """Run batch inference using a trained model.

    Parameters
    ----------
    cfg: dict
        Project configuration dictionary.
    csv_path: str
        Path to CSV file containing data to predict on.
    model_path: str, optional
        Path to a saved model (.joblib). If omitted, uses the best model
        recorded in ``models/version.txt``.
    output_path: str, optional
        Where to write the predictions CSV. Defaults to ``<csv_path>_preds.csv``.

    Returns
    -------
    Dict with keys ``model_path`` and ``predictions_path``.
    """
    inp = Path(csv_path)
    if not inp.exists():
        raise FileNotFoundError(f"Input CSV not found: {inp}")
    if inp.suffix.lower() != ".csv":
        raise ValueError(f"Input file must be a CSV: {inp}")

    model_file = Path(model_path) if model_path else best_model_path()
    if not model_file or not model_file.exists():
        raise FileNotFoundError(
            "Model file not found. Provide --model-path or ensure models/version.txt exists."
        )

    df = pd.read_csv(inp)
    model = joblib.load(model_file)
    X = _select_features(df, cfg)
    _validate_input(X, model)
    preds = model.predict(X)

    out = df.copy()
    out["prediction"] = preds

    # Always write to root-level output/ directory unless explicitly overridden
    if output_path:
        out_file = Path(output_path)
    else:
        out_dir = Path("output")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{inp.stem}_preds.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_file, index=False)

    return {"model_path": str(model_file), "predictions_path": str(out_file)}
