import os
import pandas as pd
from src.dataset import make_dataloader
from src.data_gen import make_federated_splits

def test_dataloader_shapes(tmp_path):
    out_dir = tmp_path / "data"
    make_federated_splits(out_dir=str(out_dir), n_clients=2, n_per_client=50)
    train_loader = make_dataloader(csv_path=os.path.join(out_dir, "client_0.csv"), batch_size=16, shuffle=True)
    x, y = next(iter(train_loader))
    assert x.ndim == 2
    assert y.ndim == 1
    assert x.shape[0] <= 16
