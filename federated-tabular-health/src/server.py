# src/server.py
"""Flower server that orchestrates FedAvg training.
Logic: The server aggregates client-provided model weights each round.
"""
from flwr.server import start_server

def main():
    # In a real run, configure strategy (FedAvg, min_fit_clients, etc.)
    start_server(server_address="0.0.0.0:8080", config={"num_rounds": 3})

if __name__ == "__main__":
    main()
