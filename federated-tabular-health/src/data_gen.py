import numpy as np, pandas as pd
from pathlib import Path

RNG = np.random.default_rng(7)

def _make_population(n, shift=0.0, p_pos=0.35):
    age = RNG.normal(55+shift, 12, n).clip(18, 90)
    bmi = RNG.normal(28+0.2*shift, 6, n).clip(16, 50)
    hr  = RNG.normal(78+0.5*shift, 12, n).clip(40, 140)
    sbp = RNG.normal(125+shift, 15, n).clip(80, 200)
    dbp = RNG.normal(78, 10, n).clip(50, 120)
    chol= RNG.normal(200+0.8*shift, 45, n).clip(100, 400)
    sex = RNG.integers(0,2,size=n)
    # outcome logits: linear combo + noise
    z = (0.03*(age-50) + 0.05*(bmi-25) + 0.02*(hr-70) +
         0.02*(sbp-120) + 0.01*(chol-180) + 0.2*sex +
         RNG.normal(0,0.8,n))
    p = 1/(1+np.exp(-(z - 0.5)))  # shift difficulty
    y = (RNG.random(n) < (0.25 + 0.5*p_pos)*p).astype(int)
    X = np.c_[age,bmi,hr,sbp,dbp,chol,sex]
    cols = ["age","bmi","hr","sbp","dbp","chol","sex"]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="label")

def make_federated_splits(out_dir="data", n_clients=5, n_per_client=400):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    meta = []
    for k in range(n_clients):
        shift = (k - n_clients/2) * 1.5  # non-IID domain shift
        X,y = _make_population(n_per_client, shift=shift, p_pos=0.25+0.1*(k%2))
        df = X.copy(); df["label"] = y
        path = Path(out_dir)/f"client_{k}.csv"
        df.to_csv(path, index=False)
        meta.append({"client":k, "rows":len(df), "shift":shift, "pos_rate":float(df.label.mean())})
    pd.DataFrame(meta).to_csv(Path(out_dir)/"clients_meta.csv", index=False)
    print(f"Wrote {n_clients} client CSVs to {out_dir}")

if __name__ == "__main__":
    make_federated_splits()
