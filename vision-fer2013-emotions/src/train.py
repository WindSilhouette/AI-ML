# src/train.py
import torch, os, argparse
from torch import nn, optim
from .data import make_loaders
from .models import SmallCNN, make_resnet18

def train_one_epoch(model, loader, opt, criterion, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x,y in loader:
        x = x.to(device)
        y = torch.as_tensor(y, device=device)
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward(); opt.step()
        loss_sum += float(loss.item()) * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x,y in loader:
        x = x.to(device)
        y = torch.as_tensor(y, device=device)
        logits = model(x); loss = criterion(logits, y)
        loss_sum += float(loss.item()) * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["smallcnn","resnet18"], default="smallcnn")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=48, help="FER is 48x48; keep consistent")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Let data loader honor img-size
    from torchvision import transforms
    from PIL import Image
    # patch data transform at runtime
    import types
    from . import data as data_mod
    def _make_loaders(root, batch_size=32, val_ratio=0.2):
        ds = data_mod.TinyFolderDataset(root)
        ds.t = transforms.Compose([transforms.Resize((args.img_size,args.img_size)), transforms.ToTensor()])
        from torch.utils.data import random_split, DataLoader
        n_val = max(1, int(len(ds)*val_ratio))
        n_train = len(ds) - n_val
        train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False),
            ds.class_to_idx
        )
    train_loader, val_loader, class_to_idx = _make_loaders("data/sample", batch_size=args.batch_size)

    n_classes = len(class_to_idx)
    if args.model == "smallcnn":
        model = SmallCNN(n_classes).to(device)
    else:
        model = make_resnet18(n_classes, grayscale=True, freeze_backbone=False).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        print(f"epoch={epoch} train_acc={tr_acc:.3f} val_acc={va_acc:.3f}")
        if va_acc > best_acc:
            best_acc = va_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/{args.model}.pt")

    print("best_val_acc=", best_acc)

if __name__ == "__main__":
    main()
