import os, torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from .data import make_test_loader
from .models import SmallCNN, make_resnet18

@torch.no_grad()
def run_eval(model_name="resnet18", data_root="data/fer", img_size=48, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader, class_to_idx = make_test_loader(root=data_root, img_size=img_size, batch_size=batch_size)
    labels = [c for c,_ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    n = len(labels)
    if model_name=="smallcnn": model, ckpt = SmallCNN(n).to(device), "models/smallcnn.pt"
    else:                      model, ckpt = make_resnet18(n, True).to(device), "models/resnet18.pt"
    assert os.path.exists(ckpt), f"Missing checkpoint: {ckpt}"
    model.load_state_dict(torch.load(ckpt, map_location=device)); model.eval()

    ys, ps = [], []
    for xb,yb in test_loader:
        xb = xb.to(device)
        ps.extend(model(xb).argmax(1).cpu().tolist())
        ys.extend(yb.tolist())

    print(classification_report(ys, ps, target_names=labels, digits=4))
    cm = confusion_matrix(ys, ps, labels=list(range(n)))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig = disp.plot(xticks_rotation="vertical").figure_
    os.makedirs("artifacts", exist_ok=True)
    fig.tight_layout(); fig.savefig("artifacts/confusion_matrix_test.png", dpi=160)
    print("Saved artifacts/confusion_matrix_test.png")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["smallcnn","resnet18"], default="resnet18")
    ap.add_argument("--data-root", type=str, default="data/fer")
    ap.add_argument("--img-size", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    run_eval(args.model, args.data_root, args.img_size, args.batch_size)
