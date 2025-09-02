import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

# Expected feature order for synthetic health data
DEFAULT_FEATURES = ["age", "bmi", "hr", "sbp", "dbp", "chol", "sex"]
LABEL_COL = "label"


def _load_xy(csv_path: str):
    df = pd.read_csv(csv_path)

    # --- Find label column ---
    label_col = None
    for alt in [LABEL_COL, "y", "target", "Label"]:
        if alt in df.columns:
            label_col = alt
            break
    if label_col is None:
        raise ValueError(f"No label/target column found in {csv_path}")

    # --- Pick feature columns ---
    feats = [c for c in DEFAULT_FEATURES if c in df.columns]
    if not feats:  # fallback: all numeric except label
        feats = [c for c in df.select_dtypes(include=["number"]).columns if c != label_col]

    # --- Convert to tensors ---
    X = torch.tensor(df[feats].values, dtype=torch.float32)
    y = torch.tensor(df[label_col].values, dtype=torch.long)

    return X, y


def make_dataloader(
    csv_path: str, batch_size: int = 64, shuffle: bool = True, num_workers: int = 0
) -> DataLoader:
    """
    Create a PyTorch DataLoader from a CSV file.

    The CSV should contain a label column (default 'label') and feature columns.
    If the default feature set is not present, all numeric columns except the label are used.
    """
    X, y = _load_xy(csv_path)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dl
