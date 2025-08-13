# src/cli.py
import argparse, yaml, os, subprocess, venv, platform, shutil
from pathlib import Path
from typing import Optional, Dict, Any, List

def load_cfg(path: str | Path) -> Dict[str, Any]:
    """Load config from a directory of YAML files or a single YAML file.

    If ``path`` is a directory, expects files dataset.yaml, model.yaml,
    train.yaml, and eval.yaml. Returns a dict with those top-level keys.
    """
    p = Path(path)
    if p.is_dir():
        files = ["dataset.yaml", "model.yaml", "train.yaml", "eval.yaml"]
        cfg: Dict[str, Any] = {}
        for f in files:
            fpath = p / f
            if fpath.exists():
                cfg[f.split(".")[0]] = yaml.safe_load(open(fpath))
            else:
                cfg[f.split(".")[0]] = {}
        return cfg
    return yaml.safe_load(open(p))


def save_cfg(cfg: Dict[str, Any], path: str | Path) -> Dict[str, Any]:
    """Save config to a directory of YAML files or a single YAML file.

    - If ``path`` is a directory, writes dataset.yaml, model.yaml,
      train.yaml, eval.yaml from keys in ``cfg``. Missing keys write
      empty documents.
    - If ``path`` is a file, dumps the provided dict to that file.
    Returns a dict summarizing written files.
    """
    p = Path(path)
    written: List[str] = []
    p.mkdir(parents=True, exist_ok=True) if p.is_dir() else p.parent.mkdir(parents=True, exist_ok=True)
    if p.is_dir():
        for name in ["dataset", "model", "train", "eval"]:
            f = p / f"{name}.yaml"
            with open(f, "w") as fh:
                yaml.safe_dump(cfg.get(name, {}) or {}, fh, sort_keys=False)
            written.append(str(f))
    else:
        with open(p, "w") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)
        written.append(str(p))
    return {"ok": True, "written": written}

def ensure_dirs(cfg) -> Dict[str, Any]:
    paths = cfg["dataset"]["paths"]
    created = []
    for k in ["raw", "interim", "processed"]:
        d = Path(paths[k])
        d.mkdir(parents=True, exist_ok=True)
        created.append(str(d))
    for d in [Path("models/artifacts"), Path("models/metrics"), Path("reports/figures")]:
        d.mkdir(parents=True, exist_ok=True)
        created.append(str(d))
    return {"ok": True, "created": created}

def cmd_prepare(cfg) -> Dict[str, Any]:
    res = ensure_dirs(cfg)
    return {"ok": True, "message": "Created data, models, reports dirs", **res}

def _project_root() -> Path:
    # Assumes this file lives at <project>/src/cli.py
    return Path(__file__).resolve().parents[1]

def _venv_paths(venv_dir: Path):
    if platform.system() == "Windows":
        python = venv_dir / "Scripts" / "python.exe"
        pip = venv_dir / "Scripts" / "pip.exe"
        activate = venv_dir / "Scripts" / "activate.bat"
    else:
        python = venv_dir / "bin" / "python"
        pip = venv_dir / "bin" / "pip"
        activate = venv_dir / "bin" / "activate"
    return python, pip, activate

def cmd_init_venv(spawn_shell: bool = False) -> Dict[str, Any]:
    root = _project_root()
    venv_dir = root / ".venv"
    req_file = root / "requirements.txt"
    notes: List[str] = []

    # Create or reuse the venv
    if not venv_dir.exists():
        notes.append("Creating virtual environment...")
        venv.create(venv_dir, with_pip=True)
    else:
        notes.append("Reusing existing virtual environment.")

    python, pip, activate = _venv_paths(venv_dir)
    if not python.exists():
        raise RuntimeError("Python executable not found in venv.")

    # On Apple Silicon Macs, ensure OpenMP runtime for XGBoost
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        brew = shutil.which("brew")
        if brew:
            try:
                subprocess.check_call([brew, "list", "libomp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                notes.append("Homebrew libomp already installed.")
            except subprocess.CalledProcessError:
                notes.append("Installing libomp via Homebrew for XGBoost...")
                subprocess.check_call([brew, "install", "libomp"])
        else:
            notes.append("Homebrew not found. Install libomp manually: 'brew install libomp'.")

    # Upgrade pip and essential build tools
    notes.append("Upgrading pip, setuptools, wheel...")
    subprocess.check_call([str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    # Install requirements if present
    if req_file.exists():
        notes.append(f"Installing requirements from {req_file}...")
        subprocess.check_call([str(pip), "install", "-r", str(req_file)])
    else:
        notes.append("No requirements.txt found; skipping dependency install.")

    summary = {
        "ok": True,
        "root": str(root),
        "venv_dir": str(venv_dir),
        "python": str(python),
        "pip": str(pip),
        "activate": str(activate),
        "notes": notes,
    }

    # Optionally spawn an activated shell
    if spawn_shell:
        if platform.system() == "Windows":
            # Launch cmd with venv activated
            cmd = ["cmd.exe", "/k", str(activate)]
            os.execvp(cmd[0], cmd)
        else:
            # Spawn interactive bash/zsh with venv activated
            shell = shutil.which(os.environ.get("SHELL", "bash")) or "/bin/bash"
            os.execvp(shell, [shell, "-i", "-c", f"source '{activate}'; exec {shell} -i"])  
    return summary

def _load_env_from_dotenv(dotenv_path: Path) -> Dict[str, Any]:
    try:
        from dotenv import load_dotenv
    except Exception:
        return {"ok": False, "loaded": False, "message": "python-dotenv not installed"}
    if dotenv_path.exists():
        load_dotenv(dotenv_path)
        return {"ok": True, "loaded": True, "path": str(dotenv_path)}
    else:
        return {"ok": True, "loaded": False, "path": str(dotenv_path)}

def _get_kaggle_dataset_slug(cfg, override: Optional[str]) -> Optional[str]:
    if override:
        return override
    return (cfg.get("dataset", {}).get("kaggle", {}) or {}).get("dataset")

def cmd_download_data(cfg, dataset_slug: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
    root = _project_root()
    _ = _load_env_from_dotenv(root / ".env")

    # Validate credentials
    user = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if not user or not key:
        raise RuntimeError("Missing KAGGLE_USERNAME/KAGGLE_KEY in environment. Set them in .env or your shell.")

    # Resolve dataset slug
    slug = _get_kaggle_dataset_slug(cfg, dataset_slug)
    if not slug:
        raise RuntimeError("Dataset slug not provided. Use --dataset or set dataset.kaggle.dataset in configs/dataset.yaml")

    # Ensure output dirs
    ensure_dirs(cfg)
    raw_dir = Path(cfg["dataset"]["paths"]["raw"]).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        raise ImportError("The 'kaggle' package is not installed. Activate your venv and run 'pip install kaggle'.")

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        raise RuntimeError(f"Kaggle authentication failed: {e}")

    try:
        api.dataset_download_files(dataset=slug, path=str(raw_dir), unzip=True, force=force, quiet=True)
    except Exception as e:
        raise RuntimeError(f"Kaggle download failed: {e}")
    return {"ok": True, "dataset": slug, "raw_dir": str(raw_dir)}

def cmd_build_data(cfg, raw_filename: Optional[str] = None) -> Dict[str, Any]:
    from src.data.load import load_raw_df
    from src.data.split import split_dataframe, persist_interim_splits
    from src.data.preprocess import preprocess_and_persist

    # Ensure folders exist
    ensure_dirs(cfg)
    df, path = load_raw_df(cfg, filename=raw_filename)
    df_tr, df_va, df_te = split_dataframe(df, cfg)
    persist_interim_splits(df_tr, df_va, df_te, cfg)
    preprocess_and_persist(df_tr, df_va, df_te, cfg)
    return {
        "ok": True,
        "raw_path": str(path),
        "rows": int(len(df)),
        "interim_dir": str(Path(cfg["dataset"]["paths"]["interim"]).resolve()),
        "processed_dir": str(Path(cfg["dataset"]["paths"]["processed"]).resolve()),
        "artifacts_dir": str(Path("models/artifacts").resolve()),
    }

def cmd_train(cfg) -> Dict[str, Any]:
    """Wrapper for training to keep CLI symmetry with other commands."""
    from src.training.train import train_and_save

    result = train_and_save(cfg)
    return result

def cmd_evaluate(cfg, model_path: Optional[str] = None) -> Dict[str, Any]:
    """Optional: wrapper for evaluation for consistency."""
    from src.training.evaluate import evaluate_saved_model

    out = evaluate_saved_model(cfg, model_path=model_path)
    return out


def cmd_predict(cfg, csv_path: str, model_path: Optional[str] = None, output_path: Optional[str] = None) -> Dict[str, Any]:
    """Run batch inference on a CSV file."""
    from src.inference.predict import predict_from_csv
    out = predict_from_csv(cfg, csv_path, model_path=model_path, output_path=output_path)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "command",
        choices=[
            "prepare",
            "init-venv",
            "download-data",
            "build-data",
            "train",
            "evaluate",
            "predict",
        ],
    )
    ap.add_argument("--config", default="configs")
    ap.add_argument("--input", default=None)
    ap.add_argument("--shell", action="store_true", help="After setup, spawn an activated shell (POSIX/Windows)")
    ap.add_argument("--dataset", default=None, help="Kaggle dataset slug (owner/dataset)")
    ap.add_argument("--force", action="store_true", help="Force re-download from Kaggle (overwrite)")
    ap.add_argument("--raw-filename", default=None, help="Raw CSV filename in data/raw (optional)")
    ap.add_argument("--model-path", default=None, help="Path to trained model for predict/evaluate")
    ap.add_argument("--output", default=None, help="Where to write predictions CSV (predict)")
    a = ap.parse_args()
    cfg = load_cfg(a.config)

    if a.command == "prepare":
        res = cmd_prepare(cfg)
        print(res.get("message", "prepare completed"))
    elif a.command == "init-venv":
        try:
            res = cmd_init_venv(spawn_shell=a.shell)
            print("[venv] Done. Python:", res.get("python"))
        except Exception as e:
            print(f"[venv] Error: {e}")
            raise SystemExit(1)
    elif a.command == "download-data":
        try:
            res = cmd_download_data(cfg, dataset_slug=a.dataset, force=a.force)
            print(f"[kaggle] Downloaded {res['dataset']} to {res['raw_dir']}")
        except Exception as e:
            print(f"[kaggle] Error: {e}")
            raise SystemExit(1)
    elif a.command == "build-data":
        try:
            res = cmd_build_data(cfg, raw_filename=a.raw_filename)
            print(f"[data] Built datasets from {res['raw_path']} ({res['rows']} rows)")
        except Exception as e:
            print(f"[data] Error: {e}")
            raise SystemExit(1)
    elif a.command == "train":
        try:
            res = cmd_train(cfg)
            print(f"[train] Run: {res['run_name']} | Model: {res['model_path']}")
            print(f"[train] Val: {res['val_metrics']} | Test: {res['test_metrics']}")
        except Exception as e:
            print(f"[train] Error: {e}")
            raise SystemExit(1)
    elif a.command == "evaluate":
        try:
            res = cmd_evaluate(cfg, model_path=a.model_path)
            print(f"[evaluate] Model: {res['model_path']} | Test: {res['test']}")
        except Exception as e:
            print(f"[evaluate] Error: {e}")
            raise SystemExit(1)
    elif a.command == "predict":
        if not a.input:
            print("--input CSV path required for predict")
            raise SystemExit(2)
        try:
            res = cmd_predict(cfg, csv_path=a.input, model_path=a.model_path, output_path=a.output)
            print(f"[predict] Model: {res['model_path']} | Predictions: {res['predictions_path']}")
        except Exception as e:
            print(f"[predict] Error: {e}")
            raise SystemExit(1)

if __name__ == "__main__":
    main()
