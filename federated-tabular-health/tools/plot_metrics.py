import os, sys, argparse
import pandas as pd
import matplotlib.pyplot as plt

def main(run_dir: str):
    metrics_path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        print(f"No metrics.csv in {run_dir}")
        return
    df = pd.read_csv(metrics_path)
    # One chart per metric vs round if present
    if "round" in df.columns:
        for col in df.columns:
            if col in ["round", "phase"]:
                continue
            if df[col].dtype == float or df[col].dtype == int:
                plt.figure()
                plt.plot(df["round"], df[col], marker="o")
                plt.xlabel("round")
                plt.ylabel(col)
                plt.title(f"{col} over rounds")
                out = os.path.join(run_dir, f"{col}_over_rounds.png")
                plt.savefig(out, bbox_inches="tight", dpi=160)
                plt.close()
                print(f"Saved {out}")
    else:
        print("No 'round' column; nothing to plot.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to a run folder under runs/")
    args = ap.parse_args()
    main(args.run_dir)
