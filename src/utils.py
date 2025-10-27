
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def metrics_to_df(metrics: Dict, path: str):
    df = pd.DataFrame([metrics])
    if os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)

def param_count(params: Tuple[np.ndarray, ...]) -> int:
    return sum(p.size for p in params)

def numpy_weights(model):
    return tuple(p.detach().cpu().numpy() for p in model.parameters())

def set_weights(model, weights):
    for p, w in zip(model.parameters(), weights):
        p.data = (p.data.new_tensor(w)).to(p.data.device)
