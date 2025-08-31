import os, argparse, torch
from torch import nn, optim
from collections import Counter
from .data import make_train_val_loaders
from .models import SmallCNN, make_resnet18

def train_one_epoch(model, loader, opt, criterion, device):
    model.train(); tot=cor=loss_sum=0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad(); logits = model(x)
        loss = criterion(logits, y); loss.backward(); opt.step()
        loss_sum += float(loss.item())*x.size(0)
        cor += (logits.argmax(1)==y).sum().item(); tot += x.size(0)
    return loss_sum/tot, cor/tot

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval(); tot=cor=loss_sum=0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x); loss = criterion(logits, y)
        loss_sum += float(loss.item())*x.size(0)
        cor += (logits.argmax(1)==y).sum().item(); tot += x.size(0)
    return loss_sum/tot, cor/tot

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["smallcnn","resnet18"], default="resnet18")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=48)
    ap.add_argument("--data-root", type=str, default="data/fer")
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, class_to_idx = make_train_val_loaders(args.data_root, args.img_size, args.batch_size)
    n = len(class_to_idx)

    counts = Counter()
    for _, y in tr:
        for yi in y.tolist(): counts[yi]+=1
    weights = torch.tensor([1.0/max(1,counts[i]) for i in range(n)], device=device, dtype=torch.float32)

    criterion = nn.CrossEntropyLoss(weight=weights)
    model = SmallCNN(n).to(device) if args.model=="smallcnn" else make_resnet18(n, True).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)  # no 'verbose' for older torch

    best=0.0; bad=0
    for epoch in range(args.epochs):
        tr_loss,tr_acc = train_one_epoch(model, tr, opt, criterion, device)
        va_loss,va_acc = evaluate(model, va, criterion, device)
        sched.step(va_acc)
        print(f"epoch={epoch} train_acc={tr_acc:.3f} val_acc={va_acc:.3f}")
        if va_acc>best:
            best=va_acc; bad=0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/{args.model}.pt")
        else:
            bad+=1
            if bad>=args.patience:
                print("Early stopping."); break
    print("best_val_acc=", best)

if __name__=="__main__":
    main()
