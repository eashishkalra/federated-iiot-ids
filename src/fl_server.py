
import argparse, pickle, json
import flwr as fl
from utils import DATA_DIR, OUT_DIR
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    args = ap.parse_args()

    strat = fl.server.strategy.FedAvg()
    fl.server.start_server(server_address="0.0.0.0:8080", strategy=strat, config=fl.server.ServerConfig(num_rounds=args.rounds))

if __name__ == "__main__":
    main()
