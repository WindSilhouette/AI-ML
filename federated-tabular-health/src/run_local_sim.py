# src/run_local_sim.py
"""Run a quick centralized baseline to compare with FL."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    X, y = make_classification(n_samples=1500, n_features=8, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1)
    clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
    print("Centralized baseline accuracy:", accuracy_score(yte, clf.predict(Xte)))

if __name__ == "__main__":
    main()
