# RAG Notes Helper (Lightweight LLM App)

**What it is**: Local retrieval‑augmented generation over your study notes. Chunk PDFs/Markdown, build FAISS index, and run a small local model via Transformers.

**Tech stack**: Python 3.11, faiss-cpu, transformers, sentence-transformers, uvicorn, fastapi, pydantic, pytest

## Why this project stands out
- Real‑world relevance and clean architecture
- Reproducible experiments + unit tests
- Clear benchmarks and ablations

## How you'd build it (quick logic explainer)
1. **Define the problem** → pick metrics that matter (accuracy, F1, AUROC, latency).
2. **Design the pipeline** → split into modular steps with typed I/O.
3. **Implement a minimal end‑to‑end path** → a tiny dataset sample processed by all stages.
4. **Instrument & test** → unit tests for each module and CI checks.
5. **Iterate** → add ablations/baselines; document tradeoffs in `REPORT.md`.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
```

## Repo structure
```
rag-notes-helper/
  ├─ src/
  │   └─ __init__.py
  ├─ tests/
  │   └─ test_smoke.py
  ├─ data/
  │   └─ sample/  # tiny sample to make the pipeline runnable
  ├─ models/      # saved weights
  ├─ artifacts/   # figs, logs, reports
  ├─ requirements.txt
  ├─ ROADMAP.md
  ├─ REPORT.md
  ├─ LICENSE
  └─ .gitignore
```

---
