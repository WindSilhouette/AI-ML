import subprocess, sys, time, argparse, yaml
from src.helpers import set_seed, new_run_dir, metrics_writer

# ---------- helpers -----------------------------------------------------------
def launch_server(server_address: str, rounds: int):
    # Start server as a separate process so we can spawn clients here
    cmd = [
        sys.executable, "-m", "src.server_flower",
        "--server-address", server_address,
        "--rounds", str(rounds),
    ]
    return subprocess.Popen(cmd)

def launch_clients(n: int, server_address: str, epochs=1, lr=1e-3, batch_size=64):
    procs = []
    for i in range(n):
        cmd = [
            sys.executable, "-m", "src.client_flower",
            "--server-address", server_address,
            "--client-id", str(i),
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--batch-size", str(batch_size),
        ]
        procs.append(subprocess.Popen(cmd))
        time.sleep(0.25)  # small stagger helps on Windows
    return procs

# ---------- main --------------------------------------------------------------
def orchestrate(rounds: int, server_address: str, n_clients: int, log_fn=None):
    # Start server first
    server_proc = launch_server(server_address, rounds)
    time.sleep(1.0)  # give server a moment to bind

    # Launch clients
    client_procs = launch_clients(n=n_clients, server_address=server_address)

    # Wait for server to finish all rounds
    server_rc = server_proc.wait()

    # Ensure clients are closed
    for p in client_procs:
        try:
            p.wait(timeout=5)
        except Exception:
            p.kill()

    if log_fn:
        log_fn({"phase": "federated_summary", "round": rounds, "server_rc": server_rc})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--server-address", type=str, default=None)
    ap.add_argument("--n-clients", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    rounds = args.rounds or cfg["federated"]["rounds"]
    server_address = args.server_address or cfg["federated"]["server_address"]
    n_clients = args.n_clients or cfg["data"]["n_clients"]

    run_dir = new_run_dir("fedavg")
    log = metrics_writer(run_dir)

    orchestrate(rounds=rounds, server_address=server_address, n_clients=n_clients, log_fn=log)
    print(f"Saved federated run to {run_dir}")
