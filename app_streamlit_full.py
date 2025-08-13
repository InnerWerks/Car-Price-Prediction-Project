import json
import os
from pathlib import Path
from typing import Dict, Any

import streamlit as st

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


def _read_yaml_file(path: Path) -> str:
    return path.read_text() if path.exists() else ""


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

# Sidebar: Project/config location and YAML editors
st.sidebar.header("Configuration")
config_dir = st.sidebar.text_input("Config directory", value="configs")
cfg_dir_path = Path(config_dir)

if "cfg" not in st.session_state or st.session_state.get("cfg_dir") != str(cfg_dir_path.resolve()):
    try:
        st.session_state.cfg = load_cfg(cfg_dir_path)
        st.session_state.cfg_dir = str(cfg_dir_path.resolve())
    except Exception as e:
        st.sidebar.error(f"Failed to load config: {e}")
        st.session_state.cfg = {"dataset": {}, "model": {}, "train": {}, "eval": {}}

cfg = st.session_state.cfg

with st.sidebar.expander("dataset.yaml", expanded=False):
    dataset_text = st.text_area("dataset.yaml", value=_read_yaml_file(cfg_dir_path / "dataset.yaml"), height=220, key="dataset_yaml")
with st.sidebar.expander("model.yaml", expanded=False):
    model_text = st.text_area("model.yaml", value=_read_yaml_file(cfg_dir_path / "model.yaml"), height=220, key="model_yaml")
with st.sidebar.expander("train.yaml", expanded=False):
    train_text = st.text_area("train.yaml", value=_read_yaml_file(cfg_dir_path / "train.yaml"), height=220, key="train_yaml")
with st.sidebar.expander("eval.yaml", expanded=False):
    eval_text = st.text_area("eval.yaml", value=_read_yaml_file(cfg_dir_path / "eval.yaml"), height=220, key="eval_yaml")

if st.sidebar.button("Save configs"):
    import yaml

    try:
        new_cfg = {
            "dataset": yaml.safe_load(dataset_text) or {},
            "model": yaml.safe_load(model_text) or {},
            "train": yaml.safe_load(train_text) or {},
            "eval": yaml.safe_load(eval_text) or {},
        }
        res = save_cfg(new_cfg, cfg_dir_path)
        st.session_state.cfg = new_cfg
        st.sidebar.success("Configs saved.")
        st.sidebar.json(res)
    except Exception as e:
        st.sidebar.error(f"Failed to save configs: {e}")


tab_setup, tab_data, tab_build, tab_train, tab_predict = st.tabs(
    ["Setup", "Data", "Build", "Train/Eval", "Predict"]
)

with tab_setup:
    st.subheader("Environment & Project Setup")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Kaggle Credentials (.env)")
        kaggle_user = st.text_input("KAGGLE_USERNAME", value=os.getenv("KAGGLE_USERNAME", ""))
        kaggle_key = st.text_input("KAGGLE_KEY", type="password", value=os.getenv("KAGGLE_KEY", ""))
        if st.button("Save .env"):
            try:
                _write_env_vars({"KAGGLE_USERNAME": kaggle_user, "KAGGLE_KEY": kaggle_key}, Path(".env"))
                st.success(".env updated.")
            except Exception as e:
                st.error(f"Failed to write .env: {e}")
    with col2:
        st.markdown("Project Folders")
        if st.button("Prepare folders (data/models/reports)"):
            try:
                res = cmd_prepare(st.session_state.cfg)
                _persist_result("prepare", res)
                st.success(res.get("message", "Prepared."))
                st.json(res)
            except Exception as e:
                st.error(str(e))
        st.caption("Virtualenv setup is available via CLI; running from Streamlit may not be desired.")
        if st.button("Init venv (advanced)"):
            try:
                res = cmd_init_venv(spawn_shell=False)
                _persist_result("init_venv", res)
                st.success("Virtualenv ready.")
                st.json(res)
            except Exception as e:
                st.error(str(e))

with tab_data:
    st.subheader("Acquire Data")
    ds_cfg = st.session_state.cfg.get("dataset", {})
    default_slug = (ds_cfg.get("kaggle", {}) or {}).get("dataset", "")
    dataset_slug = st.text_input("Kaggle dataset slug (owner/dataset)", value=default_slug)
    force = st.checkbox("Force re-download (overwrite)", value=False)
    if st.button("Download from Kaggle"):
        try:
            res = cmd_download_data(st.session_state.cfg, dataset_slug=dataset_slug or None, force=force)
            _persist_result("download_data", res)
            st.success("Download complete.")
            st.json(res)
        except Exception as e:
            st.error(str(e))

with tab_build:
    st.subheader("Build Datasets")
    raw_filename = st.text_input("Raw CSV filename in data/raw (optional)", value="")
    if st.button("Build data (split + preprocess)"):
        try:
            res = cmd_build_data(st.session_state.cfg, raw_filename=raw_filename or None)
            _persist_result("build_data", res)
            st.success("Build complete.")
            st.json(res)
        except Exception as e:
            st.error(str(e))

with tab_train:
    st.subheader("Train & Evaluate")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Train model"):
            try:
                res = cmd_train(st.session_state.cfg)
                _persist_result("train", res)
                st.success(f"Trained: {res['run_name']}")
                st.json(res)
            except Exception as e:
                st.error(str(e))
    with c2:
        model_path = st.text_input("Model path for evaluation (optional)", value="")
        if st.button("Evaluate model"):
            try:
                res = cmd_evaluate(st.session_state.cfg, model_path=model_path or None)
                _persist_result("evaluate", res)
                st.success("Evaluation complete.")
                st.json(res)
                figs = res.get("figures", {})
                if figs:
                    st.markdown("Figures")
                    fcols = st.columns(2)
                    idx = 0
                    for name, p in figs.items():
                        try:
                            img = Path(p)
                            if img.exists():
                                with fcols[idx % 2]:
                                    st.image(str(img), caption=name)
                            idx += 1
                        except Exception:
                            pass
            except Exception as e:
                st.error(str(e))

with tab_predict:
    st.subheader("Batch Predict")
    uploaded = st.file_uploader("Upload CSV for prediction", type=["csv"])
    model_path = st.text_input("Model path (leave blank to use best)", value="")
    out_path = st.text_input("Output path (optional)", value="")
    if st.button("Run predict", disabled=uploaded is None):
        if uploaded is None:
            st.warning("Please upload a CSV file.")
        else:
            # Save uploaded to a temp path under output/
            tmp_dir = Path("output/uploads")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            inp_path = tmp_dir / uploaded.name
            inp_path.write_bytes(uploaded.read())
            try:
                res = cmd_predict(st.session_state.cfg, csv_path=str(inp_path), model_path=model_path or None, output_path=out_path or None)
                _persist_result("predict", res)
                st.success("Prediction complete.")
                st.json(res)
                # Offer download
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

