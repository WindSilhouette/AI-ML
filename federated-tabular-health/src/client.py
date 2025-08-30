# src/client.py
"""Flower client simulating one clinic.
Logic: Local model trains on client's partition; sends weights back to server.
"""
import torch
from torch import nn, optim
import numpy as np
from flwr.client import NumPyClient
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class TinyNet(nn.Module):
    def __init__(self, in_dim, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 2)
        )
    def forward(self, x): return self.net(x)

class ClinicClient(NumPyClient):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.model = TinyNet(self.X.shape[1])
        self.opt = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return [p.detach().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, np_val in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(np_val)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for _ in range(3):
            logits = self.model(self.X)
            loss = self.criterion(logits, self.y)
            self.opt.zero_grad(); loss.backward(); self.opt.step()
        return self.get_parameters(config), len(self.X), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        with torch.no_grad():
            logits = self.model(self.X)
            acc = (logits.argmax(1) == self.y).float().mean().item()
        return float(0.0), len(self.X), {"accuracy": acc}

def get_synthetic_client(seed=0, n=500, in_dim=8):
    X, y = make_classification(
        n_samples=n, n_features=in_dim, n_informative=in_dim//2, random_state=seed
    )
    return X, y

def main():
    from flwr.client import start_numpy_client
    X, y = get_synthetic_client(seed=0)
    client = ClinicClient(X, y)
    start_numpy_client(server_address="0.0.0.0:8080", client=client)

if __name__ == "__main__":
    main()
