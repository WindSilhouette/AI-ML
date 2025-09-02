import torch
from flwr.client import NumPyClient, start_client
from src.dataset import make_dataloader
from src.model import MLP
from src.utils import train_one_epoch
from src.metrics import evaluate_model


def make_client_fn(csvs, device, epochs=1, lr=1e-3, batch_size=64):
    class FlowerClient(NumPyClient):
        def __init__(self, cid):
            self.cid = int(cid)
            self.csv = csvs[self.cid]
            self.device = device
            self.epochs = epochs
            self.lr = lr
            self.batch_size = batch_size
            self.model = MLP(input_dim=7, hidden=32, dropout=0.2).to(device)

        def get_parameters(self, config=None):
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        def set_parameters(self, parameters):
            state_dict = self.model.state_dict()
            for k, (key, val) in enumerate(state_dict.items()):
                state_dict[key] = torch.tensor(parameters[k])
            self.model.load_state_dict(state_dict)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train_loader = make_dataloader(self.csv, batch_size=self.batch_size, shuffle=True)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            for _ in range(self.epochs):
                train_one_epoch(self.model, train_loader, optimizer, self.device)
            return self.get_parameters(), len(train_loader.dataset), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            val_loader = make_dataloader(self.csv, batch_size=self.batch_size, shuffle=False)
            metrics = evaluate_model(self.model, val_loader, self.device)
            return float(metrics["loss"]), len(val_loader.dataset), metrics

    def client_fn(cid: str):
        return FlowerClient(cid)

    return client_fn


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--server-address", type=str, default="127.0.0.1:8080")
    ap.add_argument("--client-id", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csvs = [f"data/client_{k}.csv" for k in range(5)]
    client_fn = make_client_fn(csvs, device, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    start_client(server_address=args.server_address, client=client_fn(args.client_id))
