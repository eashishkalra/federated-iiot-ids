import pandas as pd, matplotlib.pyplot as plt
iid = pd.read_csv("outputs/federated_history_IID.csv")
non = pd.read_csv("outputs/federated_history_nonIID.csv")
plt.figure()
plt.plot(iid["round"], iid["acc"], label="FL IID")
plt.plot(non["round"], non["acc"], label="FL non-IID")
plt.xlabel("Round"); plt.ylabel("Accuracy"); plt.title("Federated: IID vs non-IID")
plt.legend(); plt.savefig("outputs/fig_fed_IID_vs_nonIID.png", bbox_inches="tight")
print("Saved outputs/fig_fed_IID_vs_nonIID.png")