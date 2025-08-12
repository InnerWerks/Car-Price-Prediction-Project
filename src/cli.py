# src/cli.py
import argparse, sys, yaml, os, subprocess, venv, platform, shutil
from pathlib import Path

def load_cfg(path):
    p = Path(path)
    if p.is_dir():
        files = ["dataset.yaml","model.yaml","train.yaml","eval.yaml"]
        cfg = {}
        for f in files:
            cfg[f.split(".")[0]] = yaml.safe_load(open(p/f))
        return cfg
    return yaml.safe_load(open(p))

def ensure_dirs(cfg):
    paths = cfg["dataset"]["paths"]
    for k in ["raw","interim","processed"]:
        Path(paths[k]).mkdir(parents=True, exist_ok=True)
    Path("models/artifacts").mkdir(parents=True, exist_ok=True)
    Path("models/metrics").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

def cmd_prepare(cfg):
    ensure_dirs(cfg)
    print("[prepare] OK – created data/ & models/ dirs")

def cmd_train(cfg):
    ensure_dirs(cfg)
    print("[train] stub – implement training loop in src/training/train.py")

def cmd_evaluate(cfg):
    ensure_dirs(cfg)
    print("[evaluate] stub – implement eval in src/training/evaluate.py")

def cmd_predict(cfg, inp):
    print(f"[predict] stub – implement batch prediction for: {inp}")

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare","train","evaluate","predict","init-venv"]) 
    ap.add_argument("--config", default="configs")
    ap.add_argument("--input", default=None)
    ap.add_argument("--shell", action="store_true", help="After setup, spawn an activated shell (POSIX/Windows)")
    a = ap.parse_args()
    cfg = load_cfg(a.config)

    if a.command == "prepare": cmd_prepare(cfg)
    elif a.command == "train": cmd_train(cfg)
    elif a.command == "evaluate": cmd_evaluate(cfg)
    elif a.command == "predict":
        if not a.input: sys.exit("predict requires --input")
        cmd_predict(cfg, a.input)
    elif a.command == "init-venv":
        cmd_init_venv(spawn_shell=a.shell)

if __name__ == "__main__":
    main()
