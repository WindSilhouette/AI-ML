
# Federated Tabular Health

> A **federated learning** demo using **Flower** + **PyTorch** on synthetic, **non‑IID** tabular health data.  
> Runs locally (safe by default), logs metrics per run, and includes tests, CI, and plotting utilities.

<p align="left">
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue"></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="License" src="https://img.shields.io/badge/License-MIT-green"></a>
  <img alt="OS" src="https://img.shields.io/badge/OS-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey">
</p>

## Project Features
- **Clear FL pipeline**: centralized vs local‑only vs FedAvg (Flower)
- **Reproducible**: seeded, config‑driven, with minimal tests and CI
- **Explainable**: simple MLP on synthetic tabular features
- **Visual**: per‑run metrics saved to `runs/` and quick plots

---

## Quickstart (Windows / PowerShell)

```powershell
# optional: create and activate a venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

# 1) Generate synthetic federated data
python -m src.data_gen

# 2) Baselines (centralized + local-only)
python -m src.run_baseline --epochs 5 --lr 1e-3 --batch-size 64

# 3) Federated training (Flower FedAvg)
python -m src.run_federated --rounds 5 --server-address 127.0.0.1:8080

# 4) Plot metrics for the latest run
$latest = Get-ChildItem .\runs -Directory | Sort-Object LastWriteTime -Descending | Select -First 1
python .\tools\plot_metrics.py --run_dir $($latest.FullName)
```

> **Note:** If Windows Firewall prompts the first time Python opens a port, click **Allow access**.  
> If 8080 is taken, use another port, e.g. `--server-address 127.0.0.1:9090`.

### macOS / Linux

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.data_gen
python -m src.run_baseline --epochs 5 --lr 1e-3 --batch-size 64
python -m src.run_federated --rounds 5 --server-address 127.0.0.1:8080

# Show plots for the newest run
python tools/plot_metrics.py --run_dir "$(ls -td runs/* | head -n1)"
```

---

## Project structure

```
configs/          # YAML configs (seed, data, train, federated)
src/              # package code
  ├── client_flower.py     # Flower NumPyClient wrapper (CLI-enabled)
  ├── server_flower.py     # Flower server + logging strategy
  ├── run_federated.py     # orchestration: spawns server + clients
  ├── run_baseline.py      # centralized + local-only baselines
  ├── dataset.py           # CSV → DataLoader (tabular)
  ├── model.py             # simple MLP
  ├── metrics.py           # eval (loss/acc/ROC-AUC)
  ├── utils.py             # training helpers
  └── helpers.py           # seeding, run dirs, metrics writer
tools/            # plotting & summaries
tests/            # pytest smoke tests
.github/workflows/ci.yml   # optional CI (lint + tests)
runs/             # per-run outputs (gitignored)
data/             # generated CSVs (gitignored)
```

---

## Configuration

`configs/default.yaml`

```yaml
seed: 42

data:
  n_clients: 5
  n_per_client: 300
  out_dir: "data"

train:
  epochs: 5
  batch_size: 64
  lr: 0.001

federated:
  rounds: 5
  server_address: "127.0.0.1:8080"  # local-only by default
```

Override via CLI if you don’t want to edit the YAML:
```bash
python -m src.data_gen --n-clients 10 --n-per-client 200
python -m src.run_baseline --epochs 10 --lr 5e-4
python -m src.run_federated --rounds 8 --server-address 127.0.0.1:9090
```

---

## Notes on Flower deprecations

You’ll see warnings about `start_server()`/`start_client()`—they’re from the **compat** API.  
This repo keeps things simple for local demos. To fully modernize later, switch to:

- `flower-superlink --insecure` (server)
- `flower-supernode --insecure --superlink 127.0.0.1:8080` (clients)

---

## Security

- The server binds to **`127.0.0.1`** by default → **local-only**.  
- No credentials or private data are included. Data is synthetic and generated locally.  
- Don’t change to `0.0.0.0` unless you know what you’re doing and have a firewall.

---

## Troubleshooting

- **Address in use / connect errors** → try a different port: `--server-address 127.0.0.1:9090`.
- **No `runs/` folder** → it appears only after training (`run_baseline` / `run_federated`) finishes.
- **TensorFlow oneDNN logs** (harmless):  
  - PowerShell:  
    ```powershell
    $env:TF_CPP_MIN_LOG_LEVEL="2"; $env:TF_ENABLE_ONEDNN_OPTS="0"
    ```
- **Killing stuck processes** (PowerShell):
  ```powershell
  Get-Process python | Stop-Process -Force
  ```

---

## Makefile (optional)

Mac/Linux users can run `make data | baseline | federated | plot | test`.  
Windows users can ignore the Makefile and use the Python commands above.

```make
make data
make baseline
make federated
make plot
make test
```

---

## License

[MIT](LICENSE)
