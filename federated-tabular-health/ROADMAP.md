# ROADMAP

**Milestone 1 — Minimal End‑to‑End (1–2 hrs)**
- [ ] Run `src/run_local_sim.py` for centralized baseline.
- [ ] Launch server: `python src/server.py`
- [ ] In two terminals, run two clients: `python src/client.py`
- [ ] Log accuracy per round; compare to baseline.

**Milestone 2 — Real Dataset**
- [ ] Use UCI Adult or Heart dataset; add a small CSV to `data/sample/`.
- [ ] Implement client partitioning (by clinic) with stratification.

**Milestone 3 — Experiments**
- [ ] Vary number of clients (3, 5, 10) and data heterogeneity (IID vs non‑IID).
- [ ] Compare FedAvg vs centralized; plot accuracy & fairness gap.

**Milestone 4 — Report**
- [ ] Summarize results in `REPORT.md` with figures from `artifacts/`.
