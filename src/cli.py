# src/cli.py
import argparse, sys, yaml
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare","train","evaluate","predict"])
    ap.add_argument("--config", default="configs")
    ap.add_argument("--input", default=None)
    a = ap.parse_args()
    cfg = load_cfg(a.config)

    if a.command == "prepare": cmd_prepare(cfg)
    elif a.command == "train": cmd_train(cfg)
    elif a.command == "evaluate": cmd_evaluate(cfg)
    elif a.command == "predict":
        if not a.input: sys.exit("predict requires --input")
        cmd_predict(cfg, a.input)

if __name__ == "__main__":
    main()
