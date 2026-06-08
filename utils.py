import os, json
from pathlib import Path
import joblib

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_json(d: dict, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)

def save_model(model, path: str):
    ensure_dir(os.path.dirname(path))
    joblib.dump(model, path)

def load_model(path: str):
    return joblib.load(path)
