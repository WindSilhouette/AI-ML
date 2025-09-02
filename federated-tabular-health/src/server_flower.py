import argparse
import flwr as fl

# ---- Strategy factory (simple FedAvg) ---------------------------------------
def get_strategy():
    # You can customize fit/evaluate configs or metrics aggregation later
    return fl.server.strategy.FedAvg(
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )

# ---- Run a Flower server -----------------------------------------------------
def main(server_address: str = "127.0.0.1:8080", rounds: int = 5):
    strategy = get_strategy()
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-address", type=str, default="127.0.0.1:8080")
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()
    main(server_address=args.server_address, rounds=args.rounds)
