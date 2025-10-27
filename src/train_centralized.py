
import argparse, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pathlib import Path
from utils import DATA_DIR, OUT_DIR, metrics_to_df
from model import MLP
from tqdm import tqdm

def train(model, loader, optim, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optim.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    logits_list, y_list = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            logits_list.append(logits.cpu())
            y_list.append(yb)
    logits = torch.cat(logits_list, dim=0)
    y_true = torch.cat(y_list, dim=0).numpy()
    y_pred = logits.argmax(dim=1).numpy()
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return acc, pr, rc, f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test_size", type=float, default=0.2)
    args = ap.parse_args()

    meta = json.load(open(DATA_DIR / "meta.json"))
    n_features = meta["n_features"]

    # Load all client train data to emulate centralized training
    Xs, ys = [], []
    for p in sorted(Path(DATA_DIR).glob("client_*.pkl")):
        d = pickle.load(open(p, "rb"))
        Xs.append(d["X"])
        ys.append(d["y"])
    X_tr = np.vstack(Xs)
    y_tr = np.concatenate(ys)
    test = pickle.load(open(DATA_DIR / "test.pkl", "rb"))
    X_te, y_te = test["X"], test["y"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_features=n_features, hidden=args.hidden).to(device)

    tr_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                         torch.tensor(y_tr, dtype=torch.long)),
                           batch_size=args.batch, shuffle=True)
    te_loader = DataLoader(TensorDataset(torch.tensor(X_te, dtype=torch.float32),
                                         torch.tensor(y_te, dtype=torch.long)),
                           batch_size=4096, shuffle=False)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    hist = []
    for ep in tqdm(range(1, args.epochs+1)):
        loss = train(model, tr_loader, optim, device)
        acc, pr, rc, f1 = evaluate(model, te_loader, device)
        hist.append({"epoch": ep, "loss": loss, "acc": acc, "precision": pr, "recall": rc, "f1": f1})

    # Save metrics
    pd.DataFrame(hist).to_csv(OUT_DIR / "centralized_history.csv", index=False)

    # Final metrics
    final = hist[-1]
    metrics_to_df({
        "mode": "centralized",
        "acc": final["acc"],
        "precision": final["precision"],
        "recall": final["recall"],
        "f1": final["f1"]
    }, str(OUT_DIR / "summary.csv"))

    torch.save(model.state_dict(), OUT_DIR / "centralized_mlp.pt")
    print("Saved centralized results to outputs/")

if __name__ == "__main__":
    import json
    main()
