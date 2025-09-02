import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """
    Evaluate a classification model on a dataloader.

    Returns a dict with:
      - loss: mean cross-entropy loss
      - val_acc: accuracy
      - val_auc: ROC-AUC (if both classes present; else None)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    all_probs = []
    all_targets = []

    for xb, yb in dataloader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        total_loss += loss.item() * yb.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == yb).sum().item()
        total_examples += yb.size(0)

        # collect probs for AUC if logits have >=2 classes
        if logits.shape[1] >= 2:
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_targets.extend(yb.detach().cpu().numpy().tolist())

    mean_loss = total_loss / max(1, total_examples)
    acc = total_correct / max(1, total_examples)

    # Compute ROC-AUC only if we have at least one of each class
    auc = None
    if len(all_targets) > 0:
        uniq = set(all_targets)
        if len(uniq) >= 2:
            try:
                auc = float(roc_auc_score(all_targets, all_probs))
            except Exception:
                auc = None

    return {"loss": float(mean_loss), "val_acc": float(acc), "val_auc": auc}
