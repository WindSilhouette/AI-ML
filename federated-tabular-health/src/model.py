import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim=7, hidden=64, out_dim=2, pdrop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(pdrop),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(pdrop),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)
