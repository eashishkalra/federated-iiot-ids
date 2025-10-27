
# IIoT Federated IDS — Repro Pack (Flower + PyTorch)

This project reproduces the experiments for "Federated Learning-Based Intrusion Detection in Industrial IoT Networks".
It includes:
- **Centralized baseline** training and evaluation
- **Federated learning** with Flower (FedAvg) across simulated clients
- **Non-IID partitioning** to mimic device heterogeneity
- Metric logging to CSV + plots (accuracy, precision, recall, F1; communication cost proxy)

> Tested with Python 3.11. Use a fresh virtual environment.

## 1) Setup

```bash
# create venv with CPython 3.11
py -3.11 -m venv .venv

# activate (PowerShell)
& .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## 2) Generate synthetic IIoT traffic

Create a binary classification dataset with IIoT-like features and save client partitions (IID or non-IID).

```bash
# IID partition (balanced per client)
python src/generate_data.py --clients 10 --samples 50000 --n-features 32 --non_iid 0.0

# Non-IID partition (e.g., 0.6 = strong skew)
python src/generate_data.py --clients 10 --samples 50000 --n-features 32 --non_iid 0.6
```

Outputs appear under `data/`.

## 3) Train centralized baseline

```bash
python src/train_centralized.py --test_size 0.2 --hidden 64 --epochs 8 --batch 256 --lr 1e-3
```

Metrics CSV and model checkpoint saved under `outputs/`.

## 4) Run Federated Learning (FedAvg)

### Option A: Single-process (server + sequential clients for quick demo)
```bash
python src/run_fl.py --rounds 10 --clients 10 --local_epochs 1 --batch 256 --hidden 64 --lr 1e-3
```

### Option B: Multi-process (server + N clients)
Open three terminals:
1) Server:
```bash
python src/fl_server.py --rounds 10
```
2) Clients (start as many as you have partitions, e.g., 10):
```bash
# in each client terminal (use different client_id each time)
python src/fl_client.py --client_id 0
python src/fl_client.py --client_id 1
...
```

## 5) Evaluate + Plots

```bash
# Compare final centralized vs federated metrics, and plot curves
python src/plot_results.py
```

Figures saved to `outputs/fig_*.png`.

## 6) Repro Notes

- **Model**: Lightweight MLP (PyTorch), suitable for edge.
- **Metrics**: accuracy, precision, recall, F1 (macro); plus a simple communication cost proxy (sum of parameter sizes per round).
- **Non-IID**: Dirichlet(alpha) partitioning; lower alpha => higher heterogeneity.
