
# Single-process Flower run: starts a server then simulates sequential clients per round.
import argparse, pickle, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import flwr as fl
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from model import MLP
from utils import DATA_DIR, OUT_DIR, numpy_weights, set_weights, metrics_to_df, param_count
from tqdm import tqdm

def get_loaders(X, y, batch):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch, shuffle=True)

class NumPyClient(fl.client.NumPyClient):
    def __init__(self, cid, X, y, n_features, hidden, lr, local_epochs, batch):
        self.cid = cid
        self.model = MLP(n_features, hidden=hidden)
        self.lr = lr
        self.local_epochs = local_epochs
        self.batch = batch
        self.tr_loader = get_loaders(X, y, batch)
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
        self.set_parameters(parameters)
        return 0.0, len(self.tr_loader.dataset), {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--local_epochs", type=int, default=1)
    args = ap.parse_args()

    meta = json.load(open(DATA_DIR / "meta.json"))
    n_features = meta["n_features"]
    test = pickle.load(open(DATA_DIR / "test.pkl", "rb"))
    X_te, y_te = test["X"], test["y"]
    te_loader = DataLoader(TensorDataset(torch.tensor(X_te, dtype=torch.float32),
                                         torch.tensor(y_te, dtype=torch.long)),
                           batch_size=4096, shuffle=False)

    # Load client partitions
    parts = []
    for i in range(args.clients):
        d = pickle.load(open(DATA_DIR / f"client_{i}.pkl", "rb"))
        parts.append(d)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = MLP(n_features, hidden=args.hidden).to(device)

    # Metrics storage
    history = []

    # Initial parameters
    params = [p.detach().cpu().numpy() for p in global_model.state_dict().values()]

    for rnd in tqdm(range(1, args.rounds+1)):
        client_params = []
        client_sizes = []
        # Simulate clients sequentially (quick demo). For parallelism, use fl.server.start_server + separate clients.
        for i, part in enumerate(parts):
            client = NumPyClient(i, part["X"], part["y"], n_features, args.hidden, args.lr, args.local_epochs, args.batch)
            p_upd, n, _ = client.fit(params, config={})
            client_params.append(p_upd)
            client_sizes.append(n)

        # FedAvg
        total = sum(client_sizes)
        new_params = []
        for layer_i in range(len(client_params[0])):
            agg = sum((client_sizes[j]/total) * client_params[j][layer_i] for j in range(len(client_params)))
            new_params.append(agg)

        # Set new global params
        sd = global_model.state_dict()
        for (k, _), npv in zip(sd.items(), new_params):
            sd[k] = torch.tensor(npv)
        global_model.load_state_dict(sd)

        # Evaluate global model
        global_model.eval()
        with torch.no_grad():
            logits = []
            ys = []
            for xb, yb in te_loader:
                xb = xb.to(device)
                lo = global_model(xb)
                logits.append(lo.cpu())
                ys.append(yb)
            logits = torch.cat(logits, dim=0)
            y_true = torch.cat(ys, dim=0).numpy()
            y_pred = logits.argmax(dim=1).numpy()
        acc = (y_true == y_pred).mean()
        from sklearn.metrics import precision_recall_fscore_support
        pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

        # Simple communication proxy (sum of parameter sizes broadcast + collected)
        layer_sizes = [np.prod(w.shape) for w in new_params]
        comm_this_round = 2 * sum(layer_sizes) * 8  # bytes (float64) proxy

        history.append({"round": rnd, "acc": acc, "precision": pr, "recall": rc, "f1": f1, "comm_bytes": comm_this_round})

    pd.DataFrame(history).to_csv(OUT_DIR / "federated_history.csv", index=False)
    torch.save(global_model.state_dict(), OUT_DIR / "federated_mlp.pt")

    # Final metrics
    final = history[-1]
    metrics_to_df({
        "mode": "federated",
        "acc": final["acc"],
        "precision": final["precision"],
        "recall": final["recall"],
        "f1": final["f1"]
    }, str(OUT_DIR / "summary.csv"))
    print("Saved federated results to outputs/")

if __name__ == "__main__":
    main()
