import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os

class TinyFolderDataset(Dataset):
    """Tiny folder structure: data/sample/<label>/*.png for quick smoke runs."""
    def __init__(self, root):
        self.items = []
        self.labels = []
        classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        for c in classes:
            d = os.path.join(root, c)
            for fname in os.listdir(d):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.items.append(os.path.join(d, fname))
                    self.labels.append(self.class_to_idx[c])
        self.t = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])

    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        img = Image.open(self.items[idx]).convert('L')  # FER is grayscale; keep simple here
        return self.t(img), self.labels[idx]

def make_loaders(root, batch_size=32, val_ratio=0.2):
    ds = TinyFolderDataset(root)
    n_val = max(1, int(len(ds)*val_ratio))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        ds.class_to_idx
    )
