import torch
from sklearn.metrics import roc_auc_score

def eval_loop(model, loader, device):
    model.eval(); correct=0; total=0; ys=[]; ps=[]
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            prob1 = logits.softmax(1)[:,1].cpu().numpy()
            ys.extend(y.cpu().numpy()); ps.extend(prob1)
            correct += (pred==y).sum().item(); total += y.size(0)
    acc = correct/total
    try:
        auc = roc_auc_score(ys, ps)
    except Exception:
        auc = float("nan")
    return acc, auc
