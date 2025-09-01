import flwr as fl

def start_server(rounds=5):
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2,
        evaluate_metrics_aggregation_fn=lambda results: {
            "acc": sum(r[1]["acc"] for r in results)/len(results),
            "auc": sum(r[1]["auc"] for r in results)/len(results),
        }
    )
    fl.server.start_server(server_address="0.0.0.0:8080", strategy=strategy, config=fl.server.ServerConfig(num_rounds=rounds))

if __name__ == "__main__":
    start_server()
