import flwr as fl, torch, torch.nn as nn
from .dataset import make_loaders
from .model import MLP
from .metrics import eval_loop
from .utils import train_one_epoch

class FLClient(fl.client.NumPyClient):
    def __init__(self, csv_path, device, epochs=1, lr=1e-3, batch_size=64):
        self.device = device
        self.train_loader, self.val_loader = make_loaders(csv_path, batch_size=batch_size, val_frac=0.2)
        self.model = MLP().to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.model.state_dict().values()]

    def set_parameters(self, params):
        sd = self.model.state_dict()
        for (k,_), w in zip(sd.items(), params):
            sd[k] = torch.tensor(w)
        self.model.load_state_dict(sd, strict=True)

    def fit(self, params, config):
        self.set_parameters(params)
        for _ in range(self.epochs):
            train_one_epoch(self.model, self.train_loader, self.opt, self.device, self.criterion)
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, params, config):
        self.set_parameters(params)
        acc, auc = eval_loop(self.model, self.val_loader, self.device)
        loss = 1-acc
        return float(loss), len(self.val_loader.dataset), {"acc": float(acc), "auc": float(auc)}

def make_client_fn(csv_paths, device, epochs=1, lr=1e-3, batch_size=64):
    def client_fn(cid: str):
        k = int(cid)
        return FLClient(csv_paths[k], device, epochs=epochs, lr=lr, batch_size=batch_size)
    return client_fn
