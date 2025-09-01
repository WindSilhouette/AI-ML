import pandas as pd, torch
from torch.utils.data import Dataset, DataLoader, random_split

FEATURES = ["age","bmi","hr","sbp","dbp","chol","sex"]

class TabularDS(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.X = torch.tensor(df[FEATURES].values, dtype=torch.float32)
        self.y = torch.tensor(df["label"].values, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def make_loaders(csv_path, batch_size=64, val_frac=0.2):
    ds = TabularDS(csv_path)
    n_val = int(len(ds)*val_frac)
    n_train = len(ds)-n_val
    tr, va = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    return (DataLoader(tr, batch_size=batch_size, shuffle=True),
            DataLoader(va, batch_size=batch_size, shuffle=False))
