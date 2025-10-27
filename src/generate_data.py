
import argparse
import numpy as np
import json
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from utils import DATA_DIR
import pickle

def dirichlet_partition(y, n_clients, alpha):
    # Returns list of indices per client using Dirichlet distribution (non-IID)
    n_classes = len(np.unique(y))
    idx_per_class = [np.where(y == c)[0] for c in range(n_classes)]
    client_indices = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        idxs = idx_per_class[c]
        if len(idxs) == 0:
            continue
        proportions = np.random.dirichlet(alpha=[alpha]*n_clients)
        proportions = (proportions / proportions.sum())
        split_points = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        splits = np.split(idxs, split_points)
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())
    return [np.array(sorted(ci)) for ci in client_indices]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--samples", type=int, default=50000)
    ap.add_argument("--n-features", type=int, default=32)
    ap.add_argument("--informative", type=int, default=20)
    ap.add_argument("--redundant", type=int, default=4)
    ap.add_argument("--non_iid", type=float, default=0.0, help="0.0 = IID; smaller => more skew if using Dirichlet")
    ap.add_argument("--test_size", type=float, default=0.2)
    args = ap.parse_args()

    X, y = make_classification(
        n_samples=args.samples,
        n_features=args.n_features,
        n_informative=args.informative,
        n_redundant=args.redundant,
        n_classes=2,
        weights=[0.6, 0.4],
        class_sep=1.25,
        random_state=42
    )

    # Global train/test split to keep a consistent test set
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_size, random_state=123, stratify=y)

    # Partition training among clients
    n_clients = args.clients
    partitions = []
    if args.non_iid and args.non_iid > 0.0:
        idx = np.arange(len(y_tr))
        client_idx = dirichlet_partition(y_tr, n_clients=n_clients, alpha=args.non_iid)
        for i in range(n_clients):
            partitions.append({
                "X": X_tr[client_idx[i]],
                "y": y_tr[client_idx[i]],
            })
    else:
        # IID: simple split
        splits = np.array_split(np.arange(len(y_tr)), n_clients)
        for sp in splits:
            partitions.append({
                "X": X_tr[sp],
                "y": y_tr[sp],
            })

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Save global test set
    with open(DATA_DIR / "test.pkl", "wb") as f:
        pickle.dump({"X": X_te, "y": y_te}, f)

    # Save client partitions
    for i, part in enumerate(partitions):
        with open(DATA_DIR / f"client_{i}.pkl", "wb") as f:
            pickle.dump(part, f)

    with open(DATA_DIR / "meta.json", "w") as f:
        f.write(json.dumps({
            "n_clients": n_clients,
            "n_features": X.shape[1],
        }))

    print(f"Saved {n_clients} client partitions and test set in {DATA_DIR}")

if __name__ == "__main__":
    main()
