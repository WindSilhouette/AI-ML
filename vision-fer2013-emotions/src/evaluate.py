import torch, os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from .data import make_loaders
from .models import SmallCNN, make_resnet18

def dump_confmat(model_name="smallcnn"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_to_idx = make_loaders("data/sample", batch_size=16)
    n_classes = len(class_to_idx)

    if model_name == "smallcnn":
        model = SmallCNN(n_classes).to(device); ckpt = "models/smallcnn.pt"
    else:
        model = make_resnet18(n_classes, grayscale=True, freeze_backbone=False).to(device); ckpt = "models/resnet18.pt"
    assert os.path.exists(ckpt), f"Missing checkpoint: {ckpt} (train first)"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    ys, ps = [], []
    with torch.no_grad():
        for x,y in val_loader:
            x = x.to(device)
            logits = model(x)
            ps.extend(logits.argmax(1).cpu().tolist())
            ys.extend(list(y))

    os.makedirs("artifacts", exist_ok=True)
    labels = [c for c,_ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    cm = confusion_matrix(ys, ps, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig = disp.plot(include_values=True, xticks_rotation="vertical").figure_
    fig.tight_layout()
    fig.savefig("artifacts/confusion_matrix.png", dpi=160)
    print("Saved artifacts/confusion_matrix.png")

if __name__ == "__main__":
    dump_confmat("smallcnn")
