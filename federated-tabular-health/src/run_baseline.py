import torch, torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
from .dataset import TabularDS, make_loaders
from .model import MLP
from .utils import train_one_epoch
from .metrics import eval_loop

ROOT = Path(__file__).resolve().parents[1]

def centralized(n_clients=5, epochs=5, lr=1e-3, bs=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_list = [TabularDS(ROOT/f"data/client_{k}.csv") for k in range(n_clients)]
    ds = ConcatDataset(ds_list)
    n_val = int(0.2*len(ds))
    n_train = len(ds)-n_val
    tr, va = torch.utils.data.random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    trl = DataLoader(tr, batch_size=bs, shuffle=True); val = DataLoader(va, batch_size=bs)
    model = MLP().to(device); opt = torch.optim.Adam(model.parameters(), lr=lr); crit = nn.CrossEntropyLoss()
    for _ in range(epochs): train_one_epoch(model, trl, opt, device, crit)
    acc, auc = eval_loop(model, val, device)
    print(f"[Centralized] acc={acc:.3f} auc={auc:.3f}")

def local_only(client_id=0, epochs=5, lr=1e-3, bs=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va = make_loaders(ROOT/f"data/client_{client_id}.csv", batch_size=bs)
    model = MLP().to(device); opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs): train_one_epoch(model, tr, opt, device)
    acc, auc = eval_loop(model, va, device)
    print(f"[Local-only c{client_id}] acc={acc:.3f} auc={auc:.3f}")

if __name__=="__main__":
    centralized()
    local_only(0)
