import torch, torch.nn as nn

def train_one_epoch(model, loader, opt, device, criterion=None):
    if criterion is None: criterion = nn.CrossEntropyLoss()
    model.train(); loss_sum=0; correct=0; total=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad(); logits = model(x); loss = criterion(logits, y)
        loss.backward(); opt.step()
        loss_sum += loss.item()*x.size(0)
        correct += (logits.argmax(1)==y).sum().item(); total += y.size(0)
    return loss_sum/total, correct/total
