
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from utils import OUT_DIR

cent = pd.read_csv(OUT_DIR / "centralized_history.csv")
fed = pd.read_csv(OUT_DIR / "federated_history.csv")

# 1) Accuracy over rounds/epochs
plt.figure()
plt.plot(cent["epoch"], cent["acc"], label="Centralized (epochs)")
plt.plot(fed["round"], fed["acc"], label="Federated (rounds)")
plt.xlabel("Epoch / Round")
plt.ylabel("Accuracy")
plt.title("Accuracy: Centralized vs Federated")
plt.legend()
plt.savefig(OUT_DIR / "fig_accuracy.png", bbox_inches="tight")

# 2) Precision/Recall/F1 for federated rounds
plt.figure()
plt.plot(fed["round"], fed["precision"], label="Precision")
plt.plot(fed["round"], fed["recall"], label="Recall")
plt.plot(fed["round"], fed["f1"], label="F1")
plt.xlabel("Round")
plt.ylabel("Score")
plt.title("Federated Metrics per Round")
plt.legend()
plt.savefig(OUT_DIR / "fig_fed_metrics.png", bbox_inches="tight")

# 3) Communication proxy
plt.figure()
cum = fed["comm_bytes"].cumsum() / (1024*1024)  # MB
plt.plot(fed["round"], cum, label="Cumulative Comm (MB)")
plt.xlabel("Round")
plt.ylabel("MB")
plt.title("Communication Cost Proxy")
plt.legend()
plt.savefig(OUT_DIR / "fig_comm.png", bbox_inches="tight")

print("Saved plots to outputs/")
