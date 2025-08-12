# src/cli.py
import argparse, sys, yaml, os, subprocess, venv, platform, shutil
from pathlib import Path
from typing import Optional

def load_cfg(path):
    p = Path(path)
    if p.is_dir():
        files = ["dataset.yaml", "model.yaml", "train.yaml", "eval.yaml"]
        cfg = {}
        for f in files:
            cfg[f.split(".")[0]] = yaml.safe_load(open(p / f))
        return cfg
    return yaml.safe_load(open(p))

def ensure_dirs(cfg):
    paths = cfg["dataset"]["paths"]
    for k in ["raw", "interim", "processed"]:
        Path(paths[k]).mkdir(parents=True, exist_ok=True)
    Path("models/artifacts").mkdir(parents=True, exist_ok=True)
    Path("models/metrics").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

def cmd_prepare(cfg):
    ensure_dirs(cfg)
    print("[prepare] OK – created data/ & models/ dirs")

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

def cmd_init_venv(spawn_shell: bool = False):
    root = _project_root()
    venv_dir = root / ".venv"
    req_file = root / "requirements.txt"

    print(f"[venv] Project root: {root}")
    print(f"[venv] Target venv: {venv_dir}")

    # Create or reuse the venv
    if not venv_dir.exists():
        print("[venv] Creating virtual environment...")
        venv.create(venv_dir, with_pip=True)
    else:
        print("[venv] Reusing existing virtual environment.")

    python, pip, activate = _venv_paths(venv_dir)
    if not python.exists():
        sys.exit("[venv] Error: Python executable not found in venv.")

    # Upgrade pip and essential build tools
    print("[venv] Upgrading pip, setuptools, wheel...")
    subprocess.check_call([str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    # Install requirements if present
    if req_file.exists():
        print(f"[venv] Installing requirements from {req_file}...")
        subprocess.check_call([str(pip), "install", "-r", str(req_file)])
    else:
        print("[venv] No requirements.txt found; skipping dependency install.")

    # Summarize
    print("[venv] Done.")
    print(f"[venv] Python: {python}")
    print(f"[venv] To activate manually:")
    if platform.system() == "Windows":
        print(f"  {venv_dir}\\Scripts\\activate.bat")
    else:
        print(f"  source {activate}")

    # Optionally spawn an activated shell
    if spawn_shell:
        if platform.system() == "Windows":
            # Launch cmd with venv activated
            cmd = ["cmd.exe", "/k", str(activate)]
            print("[venv] Spawning activated cmd shell...")
            os.execvp(cmd[0], cmd)
        else:
            # Spawn interactive bash/zsh with venv activated
            shell = shutil.which(os.environ.get("SHELL", "bash")) or "/bin/bash"
            print(f"[venv] Spawning activated shell: {shell} ...")
            os.execvp(shell, [shell, "-i", "-c", f"source '{activate}'; exec {shell} -i"])  

def _load_env_from_dotenv(dotenv_path: Path) -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        print("[env] python-dotenv not installed. Proceeding without auto-loading .env.")
        return
    if dotenv_path.exists():
        load_dotenv(dotenv_path)
        print(f"[env] Loaded environment variables from {dotenv_path}")
    else:
        print(f"[env] No .env found at {dotenv_path} (skipping)")

def _get_kaggle_dataset_slug(cfg, override: Optional[str]) -> Optional[str]:
    if override:
        return override
    return (cfg.get("dataset", {}).get("kaggle", {}) or {}).get("dataset")

def cmd_download_data(cfg, dataset_slug: Optional[str] = None, force: bool = False):
    root = _project_root()
    _load_env_from_dotenv(root / ".env")

    # Validate credentials
    user = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if not user or not key:
        sys.exit("[kaggle] Missing KAGGLE_USERNAME/KAGGLE_KEY in environment. Set them in .env or your shell.")

    # Resolve dataset slug
    slug = _get_kaggle_dataset_slug(cfg, dataset_slug)
    if not slug:
        sys.exit("[kaggle] Dataset slug not provided. Use --dataset or add dataset.kaggle.dataset in configs/dataset.yaml")

    # Ensure output dirs
    ensure_dirs(cfg)
    raw_dir = Path(cfg["dataset"]["paths"]["raw"]).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"[kaggle] Downloading '{slug}' to {raw_dir} ...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        sys.exit("[kaggle] The 'kaggle' package is not installed. Activate your venv and run 'pip install kaggle'.")

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        sys.exit(f"[kaggle] Authentication failed: {e}")

    try:
        api.dataset_download_files(dataset=slug, path=str(raw_dir), unzip=True, force=force, quiet=False)
    except Exception as e:
        sys.exit(f"[kaggle] Download failed: {e}")

    print("[kaggle] Download complete.")

def cmd_build_data(cfg, raw_filename: Optional[str] = None):
    from src.data.load import load_raw_df
    from src.data.split import split_dataframe, persist_interim_splits
    from src.data.preprocess import preprocess_and_persist

    # Ensure folders exist
    ensure_dirs(cfg)

    print("[data] Loading raw dataset...")
    df, path = load_raw_df(cfg, filename=raw_filename)
    print(f"[data] Loaded {len(df):,} rows from {path}")

    print("[data] Splitting into train/val/test...")
    df_tr, df_va, df_te = split_dataframe(df, cfg)
    persist_interim_splits(df_tr, df_va, df_te, cfg)
    print("[data] Saved interim splits to data/interim/")

    print("[data] Preprocessing and writing processed datasets...")
    preprocess_and_persist(df_tr, df_va, df_te, cfg)
    print("[data] Saved processed datasets to data/processed/ and preprocessor to models/artifacts/")

def cmd_train(cfg):
    """Wrapper for training to keep CLI symmetry with other commands."""
    from src.training.train import train_and_save

    result = train_and_save(cfg)

    print(f"[train] Run: {result['run_name']}")
    print(f"[train] Model saved to: {result['model_path']}")
    print(f"[train] Metrics saved to: {result['metrics_path']}")
    print(f"[train] Val: {result['val_metrics']}")
    print(f"[train] Test: {result['test_metrics']}")
    if result.get("best_updated"):
        print("[train] Updated models/version.txt with new best model.")

    return result

def cmd_evaluate(cfg):
    """Optional: wrapper for evaluation for consistency."""
    from src.training.evaluate import evaluate_saved_model

    out = evaluate_saved_model(cfg)
    print(f"[evaluate] Model: {out['model_path']}")
    print(f"[evaluate] Test metrics: {out['test']}")
    print(f"[evaluate] Figures: {out['figures']}")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "init-venv", "download-data", "build-data", "train", "evaluate"])
    ap.add_argument("--config", default="configs")
    ap.add_argument("--input", default=None)
    ap.add_argument("--shell", action="store_true", help="After setup, spawn an activated shell (POSIX/Windows)")
    ap.add_argument("--dataset", default=None, help="Kaggle dataset slug (owner/dataset)")
    ap.add_argument("--force", action="store_true", help="Force re-download from Kaggle (overwrite)")
    ap.add_argument("--raw-filename", default=None, help="Raw CSV filename in data/raw (optional)")
    a = ap.parse_args()
    cfg = load_cfg(a.config)

    if a.command == "prepare":
        cmd_prepare(cfg)
    elif a.command == "init-venv":
        cmd_init_venv(spawn_shell=a.shell)
    elif a.command == "download-data":
        cmd_download_data(cfg, dataset_slug=a.dataset, force=a.force)
    elif a.command == "build-data":
        cmd_build_data(cfg, raw_filename=a.raw_filename)
    elif a.command == "train":
        cmd_train(cfg)
    elif a.command == "evaluate":
        cmd_evaluate(cfg)

if __name__ == "__main__":
    main()
