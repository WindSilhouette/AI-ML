import subprocess, sys, time, os
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parents[1]

def main(n_clients=5, rounds=5):
    # 1) synth data if missing
    data_dir = ROOT/"data"
    if not (data_dir/"client_0.csv").exists():
        subprocess.check_call([PY, str(ROOT/"src"/"data_gen.py")])

    # 2) start server
    server = subprocess.Popen([PY, str(ROOT/"src"/"server_flower.py")])

    # 3) launch clients
    csvs = [str(data_dir/f"client_{k}.csv") for k in range(n_clients)]
    client_cmd = [PY, "-c",
        "import torch, sys; from src.client_flower import make_client_fn;"
        "from flwr.client import start_client;"
        "csvs=sys.argv[1:]; client_fn=make_client_fn(csvs, torch.device('cpu'), epochs=1, lr=1e-3, batch_size=64);"
        "cid=str(int(sys.argv[-1].split('_')[-1])); start_client(server_address='0.0.0.0:8080', client=client_fn(cid))"
    ]

    procs=[]
    try:
        for k in range(n_clients):
            env = os.environ.copy()
            args = client_cmd + csvs + [f"client_{k}"]
            procs.append(subprocess.Popen(args, env=env))
            time.sleep(0.5)
        server.wait()
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        if server.poll() is None:
            server.terminate()

if __name__ == "__main__":
    main()
