import os, csv, time, random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def new_run_dir(prefix="exp"):
    ts=time.strftime("%Y%m%d-%H%M%S")
    path=os.path.join("runs", f"{prefix}_{ts}")
    ensure_dir(path)
    return path

def metrics_writer(run_dir: str, filename="metrics.csv"):
    ensure_dir(run_dir)
    path=os.path.join(run_dir, filename)
    def write_row(d: dict):
        exists=os.path.exists(path)
        with open(path, "a", newline="") as f:
            w=csv.DictWriter(f, fieldnames=list(d.keys()))
            if not exists:
                w.writeheader()
            w.writerow(d)
    return write_row
