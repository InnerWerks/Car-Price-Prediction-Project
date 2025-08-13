import json
import os
from pathlib import Path
from typing import Dict, Any

import streamlit as st
import itertools

from src.cli import (
    load_cfg,
    save_cfg,
    cmd_prepare,
    cmd_init_venv,
    cmd_download_data,
    cmd_build_data,
    cmd_train,
    cmd_evaluate,
    cmd_predict,
)

from src.inference.predict import best_model_info


PRIMARY_METRIC = "r2"


APP_STATE_DIR = Path("models/metrics/ui_last_run")


def _persist_result(name: str, res: Dict[str, Any]) -> None:
    try:
        APP_STATE_DIR.mkdir(parents=True, exist_ok=True)
        (APP_STATE_DIR / f"{name}.json").write_text(json.dumps(res, indent=2))
    except Exception:
        pass


def _load_persisted(name: str) -> Dict[str, Any] | None:
    p = APP_STATE_DIR / f"{name}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _write_env_vars(kv: Dict[str, str], env_path: Path) -> None:
    # Update process env
    for k, v in kv.items():
        if v is not None:
            os.environ[k] = v
    # Merge and persist to .env
    existing = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if not line or line.strip().startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                existing[k.strip()] = v.strip()
    existing.update({k: v for k, v in kv.items() if v is not None})
    body = "\n".join(f"{k}={v}" for k, v in existing.items()) + "\n"
    env_path.write_text(body)


st.set_page_config(page_title="ML Project GUI", layout="wide")
st.title("Project Control Panel")

cfg_dir_path = Path("configs")
if "cfg" not in st.session_state:
    st.session_state.cfg = load_cfg(cfg_dir_path)
    st.session_state.cfg_dir = str(cfg_dir_path.resolve())

cfg = st.session_state.cfg

if "setup_done" not in st.session_state:
    proc_dir = Path(st.session_state.cfg.get("dataset", {}).get("paths", {}).get("processed", "data/processed"))
    processed_ready = (proc_dir / "train.csv").exists()
    st.session_state.setup_done = Path(".venv").exists() and processed_ready


tab_setup, tab_train, tab_predict = st.tabs(
    ["Setup & Data", "Train/Eval", "Predict"]
)

with tab_setup:
    st.subheader("Environment, Data & Project Setup")
    ds_cfg = st.session_state.cfg.get("dataset", {})
    dataset_slug = (ds_cfg.get("kaggle", {}) or {}).get("dataset", "")
    st.write(f"Dataset: **{dataset_slug}**")
    split_cfg = ds_cfg.get("split", {})
    with st.form("setup_form"):
        st.markdown("Kaggle Credentials")
        c1, c2 = st.columns(2)
        kaggle_user = c1.text_input("KAGGLE_USERNAME", value=os.getenv("KAGGLE_USERNAME", ""))
        kaggle_key = c2.text_input("KAGGLE_KEY", type="password", value=os.getenv("KAGGLE_KEY", ""))
        st.markdown("Data Split")
        s1, s2 = st.columns(2)
        test_size = s1.number_input(
            "Test size",
            min_value=0.05,
            max_value=0.9,
            step=0.05,
            value=float(split_cfg.get("test_size", 0.2)),
        )
        val_size = s2.number_input(
            "Validation size",
            min_value=0.05,
            max_value=0.5,
            step=0.05,
            value=float(split_cfg.get("val_size", 0.1)),
        )
        submitted_setup = st.form_submit_button("Initialize project & data")
    if submitted_setup:
        try:
            with st.spinner(
                "Setting up environment, installing dependencies, downloading and building data..."
            ):
                _write_env_vars(
                    {"KAGGLE_USERNAME": kaggle_user, "KAGGLE_KEY": kaggle_key}, Path(".env")
                )
                st.session_state.cfg["dataset"].setdefault("split", {})
                st.session_state.cfg["dataset"]["split"]["test_size"] = float(test_size)
                st.session_state.cfg["dataset"]["split"]["val_size"] = float(val_size)
                save_cfg(st.session_state.cfg, cfg_dir_path)
                res_prep = cmd_prepare(st.session_state.cfg)
                res_venv = cmd_init_venv(spawn_shell=False)
                res_dl = cmd_download_data(st.session_state.cfg, dataset_slug=None, force=False)
                res_build = cmd_build_data(st.session_state.cfg, raw_filename=None)
            combined = {
                "prepare": res_prep,
                "venv": res_venv,
                "download": res_dl,
                "build": res_build,
            }
            _persist_result("setup_data", combined)
            rows = res_build.get("rows")
            st.success(f"Setup complete. Data ready ({rows} rows).")
            st.session_state.setup_done = True
        except Exception as e:
            st.error(str(e))


with tab_train:
    st.subheader("Train & Evaluate")
    disabled = not st.session_state.setup_done
    if disabled:
        st.info("Complete Setup & Data step first.")

    model_cfg = st.session_state.cfg.setdefault("model", {}).setdefault("tabular", {})
    eval_cfg = st.session_state.cfg.get("eval", {}) or {}

    MODEL_SPECS = {
        "Linear Regression": {"key": "linear", "params": {}},
        "Ridge Regression": {"key": "ridge", "params": {"alpha": {"type": "float", "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}}},
        "Lasso Regression": {"key": "lasso", "params": {"alpha": {"type": "float", "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}}},
        "Elastic Net": {"key": "elasticnet", "params": {"alpha": {"type": "float", "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}, "l1_ratio": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}}},
        "Support Vector Regression": {"key": "svr", "params": {"C": {"type": "float", "default": 1.0, "min": 0.1, "max": 100.0, "step": 0.1}, "epsilon": {"type": "float", "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}}},
        "XGBoost Regressor": {"key": "xgboost", "params": {
            "n_estimators": {"type": "int", "default": 500, "min": 1, "max": 1000, "step": 1},
            "max_depth": {"type": "int", "default": 6, "min": 1, "max": 20, "step": 1},
            "learning_rate": {"type": "float", "default": 0.05, "min": 0.0001, "max": 1.0, "step": 0.01},
            "subsample": {"type": "float", "default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05},
            "colsample_bytree": {"type": "float", "default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05},
            "reg_lambda": {"type": "float", "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1},
            "reg_alpha": {"type": "float", "default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1},
        }},
    }
    algo_labels = list(MODEL_SPECS.keys())
    key_to_label = {spec["key"]: label for label, spec in MODEL_SPECS.items()}
    cur_algo = model_cfg.get("algorithm", "xgboost")
    idx = algo_labels.index(key_to_label.get(cur_algo, "XGBoost Regressor"))
    algo_label = st.selectbox("Model", algo_labels, index=idx, disabled=disabled)
    algo_key = MODEL_SPECS[algo_label]["key"]
    model_cfg["algorithm"] = algo_key
    param_spec = MODEL_SPECS[algo_label]["params"]
    model_params = model_cfg.get("params", {})
    search_cfg = model_cfg.get("search", {})

    # ---- Evaluation UI ----
    METRIC_OPTIONS = [
        ("rmse", "RMSE", "Root Mean Squared Error — penalizes large errors"),
        ("mae", "MAE", "Mean Absolute Error — robust to outliers"),
        ("r2", "R²", "Variance explained by the model"),
        ("adj_r2", "Adjusted R²", "R² adjusted for number of predictors"),
        ("mape", "MAPE", "Mean Absolute Percentage Error"),
        ("smape", "sMAPE", "Symmetric MAPE"),
        ("medae", "Median AE", "Median Absolute Error — very robust"),
        ("explained_variance", "Explained Var", "Explained Variance Score"),
    ]
    metric_keys = [k for k, _, _ in METRIC_OPTIONS]

    PLOT_GROUPS = {
        "Performance": [
            ("pred_vs_actual", "Predicted vs Actual", "Scatter of predictions vs ground truth"),
            ("prediction_error_hist", "Prediction Error Histogram", "Distribution of ŷ − y"),
        ],
        "Diagnostics": [
            ("residuals_vs_fitted", "Residuals vs Fitted", "Detect bias/heteroscedasticity"),
            ("residual_hist", "Residual Histogram", "Residual distribution"),
            ("residual_qq", "Residual Q–Q", "Normality check for residuals"),
        ],
        "Learning / Capacity": [
            ("learning_curve", "Learning Curve", "Train/val score vs sample size"),
        ],
        "Explainability": [
            ("feature_importances", "Feature Importances", "Top features (tree/linear models)"),
            ("pdp_1d", "Partial Dependence (1D)", "Effect of single feature on prediction"),
            ("pdp_2d", "Partial Dependence (2D)", "Joint effect of two features"),
        ],
    }
    all_plot_keys = list({k for group in PLOT_GROUPS.values() for (k, _, _) in group})

    selected_metrics = ["r2", "rmse", "mae", "medae", "explained_variance"]
    selected_plots = [
        "pred_vs_actual",
        "residuals_vs_fitted",
        "residual_hist",
        "residual_qq",
    ]

    st.session_state.cfg.setdefault("train", {})["primary_metric"] = PRIMARY_METRIC

    st.session_state.cfg["eval"] = {
        "metrics": {"regression": selected_metrics},
        "plots": {k: (k in selected_plots) for k in all_plot_keys},
    }
    save_cfg(st.session_state.cfg, cfg_dir_path)

    st.markdown("### Evaluation")
    m_badges = " · ".join([next(lbl for (k, lbl, _h) in METRIC_OPTIONS if k == m) for m in selected_metrics])
    p_badges = " · ".join(selected_plots)
    st.caption(f"**Metrics:** {m_badges}")
    st.caption(f"**Plots:** {p_badges}")

    plot_label_map = {k: lbl for group, items in PLOT_GROUPS.items() for (k, lbl, _h) in items}

    strategies = ["Single run"] if not param_spec else ["Single run", "Random search", "Grid search"]
    strategy = st.selectbox(
        "Optimization Strategy",
        strategies,
        index={"single": 0, "random": 1, "grid": 2}.get(search_cfg.get("strategy", "single"), 0) if len(strategies) > 1 else 0,
    )

    def render_single(params_spec, params_cfg):
        values = {}
        cols = st.columns(max(1, min(3, len(params_spec)))) if params_spec else []
        for i, (name, spec) in enumerate(params_spec.items()):
            col = cols[i % len(cols)] if cols else st
            if spec["type"] == "int":
                val = col.number_input(name, value=int(params_cfg.get(name, spec["default"])), min_value=int(spec["min"]), step=int(spec["step"]))
            else:
                val = col.number_input(name, value=float(params_cfg.get(name, spec["default"])), min_value=float(spec["min"]), max_value=float(spec.get("max", 1e9)), step=float(spec["step"]), format="%.4f")
            values[name] = val
        return values

    def render_random(params_spec, search_params_cfg):
        ranges = {}
        for name, spec in params_spec.items():
            st.markdown(f"**{name}**")
            c = st.columns(2)
            default_range = search_params_cfg.get(name, [spec["min"], spec["max"]])
            if spec["type"] == "int":
                mn = c[0].number_input("min", min_value=int(spec["min"]), max_value=int(spec["max"]), value=int(default_range[0]), key=f"rs_{name}_min")
                mx = c[1].number_input("max", min_value=int(spec["min"]), max_value=int(spec["max"]), value=int(default_range[1]), key=f"rs_{name}_max")
            else:
                mn = c[0].number_input("min", min_value=float(spec["min"]), max_value=float(spec["max"]), value=float(default_range[0]), step=float(spec["step"]), format="%.4f", key=f"rs_{name}_min")
                mx = c[1].number_input("max", min_value=float(spec["min"]), max_value=float(spec["max"]), value=float(default_range[1]), step=float(spec["step"]), format="%.4f", key=f"rs_{name}_max")
            ranges[name] = (mn, mx)
        return ranges

    def render_grid(params_spec, search_params_cfg):
        ranges = {}
        for name, spec in params_spec.items():
            st.markdown(f"**{name}**")
            c = st.columns(3)
            defaults = search_params_cfg.get(name, {"min": spec["min"], "max": spec["max"], "step": spec["step"]})
            if spec["type"] == "int":
                mn = c[0].number_input("min", min_value=int(spec["min"]), max_value=int(spec["max"]), value=int(defaults.get("min", spec["min"])), key=f"gs_{name}_min")
                mx = c[1].number_input("max", min_value=int(spec["min"]), max_value=int(spec["max"]), value=int(defaults.get("max", spec["max"])), key=f"gs_{name}_max")
                step = c[2].number_input("step", min_value=1, value=int(defaults.get("step", spec["step"])), key=f"gs_{name}_step")
            else:
                mn = c[0].number_input("min", min_value=float(spec["min"]), max_value=float(spec["max"]), value=float(defaults.get("min", spec["min"])), step=float(spec["step"]), format="%.4f", key=f"gs_{name}_min")
                mx = c[1].number_input("max", min_value=float(spec["min"]), max_value=float(spec["max"]), value=float(defaults.get("max", spec["max"])), step=float(spec["step"]), format="%.4f", key=f"gs_{name}_max")
                step = c[2].number_input("step", min_value=float(spec["step"]), value=float(defaults.get("step", spec["step"])), step=float(spec["step"]), format="%.4f", key=f"gs_{name}_step")
            ranges[name] = {"min": mn, "max": mx, "step": step}
        return ranges

    with st.form("train_form"):
        if strategy == "Single run":
            st.markdown("Model Hyperparameters")
            params_values = render_single(param_spec, model_params)
        elif strategy == "Random search":
            runs = st.number_input("Number of runs", min_value=1, value=int(search_cfg.get("n_runs", 10)))
            ranges = render_random(param_spec, search_cfg.get("params", {}))
        else:
            ranges = render_grid(param_spec, search_cfg.get("params", {}))
        submitted_train = st.form_submit_button("Run Training", disabled=disabled)

    if submitted_train:
        errors = []
        if strategy == "Random search":
            for name, (mn, mx) in ranges.items():
                if mn > mx:
                    errors.append(f"{name} min must be <= max")
        elif strategy == "Grid search":
            for name, cfg_rng in ranges.items():
                if cfg_rng["min"] > cfg_rng["max"]:
                    errors.append(f"{name} min must be <= max")
                if cfg_rng["step"] <= 0:
                    errors.append(f"{name} step must be > 0")
        if errors:
            for e in errors:
                st.error(e)
        else:
            if strategy == "Single run":
                model_cfg["params"] = {k: (int(v) if param_spec[k]["type"] == "int" else float(v)) for k, v in params_values.items()}
                model_cfg.pop("search", None)
            elif strategy == "Random search":
                model_cfg["search"] = {
                    "strategy": "random",
                    "n_runs": int(runs),
                    "params": {k: [float(v[0]), float(v[1])] if param_spec[k]["type"] == "float" else [int(v[0]), int(v[1])] for k, v in ranges.items()},
                }
                model_cfg.pop("params", None)
            else:
                model_cfg["search"] = {
                    "strategy": "grid",
                    "params": {k: {"min": float(v["min"]) if param_spec[k]["type"] == "float" else int(v["min"]), "max": float(v["max"]) if param_spec[k]["type"] == "float" else int(v["max"]), "step": float(v["step"]) if param_spec[k]["type"] == "float" else int(v["step"])} for k, v in ranges.items()},
                }
                model_cfg.pop("params", None)
            save_cfg(st.session_state.cfg, cfg_dir_path)
            try:
                prog = st.progress(0)
                def _cb(f):
                    prog.progress(int(f * 100))
                with st.spinner("Training and evaluating model..."):
                    res_train = cmd_train(st.session_state.cfg, progress_cb=_cb)
                    res_eval = cmd_evaluate(st.session_state.cfg, model_path=None)
                prog.progress(100)
                combined = {"train": res_train, "evaluate": res_eval}
                _persist_result("train_evaluate", combined)
                st.success("Training and evaluation complete.")

                if res_train.get("best_updated"):
                    st.info(f"Best model updated based on R² = {res_train['val_metrics']['r2']:.4f}")
                else:
                    st.info(f"No improvement over existing best (R² = {res_train['val_metrics']['r2']:.4f})")

                if selected_metrics:
                    st.markdown("#### Validation Metrics")
                    v_cols = st.columns(len(selected_metrics))
                    for i, k in enumerate(selected_metrics):
                        v = res_train.get("val_metrics", {}).get(k)
                        if v is not None:
                            v_cols[i].metric(k.upper(), f"{v:.4f}")

                    st.markdown("#### Test Metrics")
                    t_cols = st.columns(len(selected_metrics))
                    for i, k in enumerate(selected_metrics):
                        v = res_eval.get("test", {}).get(k)
                        if v is not None:
                            t_cols[i].metric(k.upper(), f"{v:.4f}")

                st.caption(f"Model saved to {res_train.get('model_path')}")
                figs = res_eval.get("figures", {})
                if figs and selected_plots:
                    st.markdown("#### Figures")
                    fcols = st.columns(2)
                    idx = 0
                    for name in selected_plots:
                        p = figs.get(name)
                        if p:
                            img = Path(p)
                            if img.exists():
                                with fcols[idx % 2]:
                                    st.image(str(img), caption=plot_label_map.get(name, name))
                                idx += 1
            except Exception as e:
                st.error(str(e))
with tab_predict:
    st.subheader("Batch Predict")
    info = best_model_info()
    if info:
        if info.get("value") is not None:
            st.caption(
                f"Best model: {info['model_path']} (validation R² = {info['value']:.4f})"
            )
        else:
            st.caption(f"Best model: {info['model_path']}")
        model_path = info["model_path"]
    else:
        st.info("Train a model to enable prediction.")
        model_path = None
    disabled = model_path is None
    uploaded = st.file_uploader("Upload CSV for prediction", type=["csv"], disabled=disabled)
    if st.button("Run predict", disabled=(uploaded is None or disabled)):
        if uploaded is None:
            st.warning("Please upload a CSV file.")
        else:
            tmp_dir = Path("output/uploads")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            inp_path = tmp_dir / uploaded.name
            inp_path.write_bytes(uploaded.read())
            try:
                with st.spinner("Running prediction..."):
                    res = cmd_predict(
                        st.session_state.cfg,
                        csv_path=str(inp_path),
                        model_path=None,
                        output_path=None,
                    )
                _persist_result("predict", res)
                st.success("Prediction complete.")
                st.caption(f"Model used: {res.get('model_path')}")
                pred_p = Path(res.get("predictions_path", ""))
                if pred_p.exists():
                    st.download_button(
                        label="Download predictions CSV",
                        data=pred_p.read_bytes(),
                        file_name=pred_p.name,
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(str(e))

