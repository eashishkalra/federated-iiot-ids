
import argparse, pickle, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import flwr as fl
from pathlib import Path
from model import MLP
from utils import DATA_DIR

def get_loader(X, y, batch):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch, shuffle=True)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, X, y, n_features, hidden, lr, local_epochs, batch):
        self.model = MLP(n_features, hidden=hidden)
        self.lr = lr
        self.local_epochs = local_epochs
        self.batch = batch
        self.tr_loader = get_loader(X, y, batch)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_parameters(self, config):
        return [p.cpu().numpy() for p in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        sd = self.model.state_dict()
        for (k, v), npv in zip(sd.items(), parameters):
            sd[k] = torch.tensor(npv)
        self.model.load_state_dict(sd)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(self.local_epochs):
            for xb, yb in self.tr_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
        return self.get_parameters(config={}), len(self.tr_loader.dataset), {}

    def evaluate(self, parameters, config):
        # Optional: implement local eval
        return 0.0, len(self.tr_loader.dataset), {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client_id", type=int, required=True)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--server", type=str, default="0.0.0.0:8080")
    args = ap.parse_args()

    meta = json.load(open(DATA_DIR / "meta.json"))
    n_features = meta["n_features"]

    part = pickle.load(open(DATA_DIR / f"client_{args.client_id}.pkl", "rb"))
    client = FlowerClient(part["X"], part["y"], n_features, args.hidden, args.lr, args.local_epochs, args.batch)
    fl.client.start_numpy_client(server_address=args.server, client=client)

if __name__ == "__main__":
    main()
