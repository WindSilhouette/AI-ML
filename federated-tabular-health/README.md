# Federated Tabular Health

A federated learning project that simulates multiple hospitals collaboratively training a model on **synthetic patient data** without sharing raw records.  
Built with **PyTorch** + **Flower (FL)**.

---

## 🚀 Features
- Synthetic health dataset generator (age, BMI, vitals, labs, sex → binary outcome).
- Federated setup with **non-IID client splits**.
- Baselines: centralized (upper bound) vs local-only training.
- Federated training with **FedAvg** aggregation.
- Extendable with **Differential Privacy** (Opacus).

---

## 📂 Project Structure
federated-tabular-health/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
├── data_gen.py # synthetic dataset generator
├── dataset.py # PyTorch Dataset + loaders
├── model.py # MLP for tabular data
├── metrics.py # accuracy & AUC
├── utils.py # training loop helper
├── client_flower.py # client logic for Flower
├── server_flower.py # server logic for Flower
├── run_federated.py # orchestrates FL run
└── run_baselines.py # centralized & local-only baselines


---

## ⚡ Quickstart

### 1. Setup environment
```bash
# Create and activate venv
python -m venv .venv
.venv\Scripts\activate   # Windows PowerShell
# OR: source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

python -m src.data_gen
python -m src.run_federated
python -m src.run_baselines

