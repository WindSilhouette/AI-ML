# RAG Notes Helper (Lightweight LLM App)

**What it is**: A local **retrieval-augmented search** tool for your study notes.  
It chunks `.pdf` / `.md` / `.txt` files, builds a **FAISS** index with **MiniLM embeddings**, and lets you query them quickly from the CLI or via a simple API.  
No external services — all local.

**Tech stack**: Python 3.10+, faiss-cpu, sentence-transformers, pypdf, pytest, (optional: FastAPI + Uvicorn)

---

## Why this project stands out
- **Real-world relevance** → organize and query your notes instantly  
- **Clean, modular architecture** → ingestion, index, query are separate modules  
- **Reproducible** → deterministic pipeline + smoke tests  
- **Extensible** → optional FastAPI endpoint, future LLM integration  

---

## How it works
1. **Ingest notes** → read PDFs/Markdown/TXT, normalize, chunk into passages  
2. **Embed & index** → generate embeddings with `all-MiniLM-L6-v2` and store in FAISS  
3. **Query** → embed a question, retrieve top-k passages, show context + draft answer  
4. **Iterate** → expand dataset, evaluate retrieval quality, integrate summarization/LLM  

---

## Setup
```bash
# Create a virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run smoke tests
pytest -q
```

---

## Repo structure
```
rag-notes-helper/
  ├─ src/
  │   ├─ build_index.py   # build FAISS index from notes
  │   ├─ run.py           # query interface (CLI)
  │   └─ api.py           # optional FastAPI app
  ├─ notes/               # your notes (.md/.txt/.pdf)
  ├─ artifacts/index/     # auto-generated FAISS index + metadata
  ├─ tests/
  │   └─ test_smoke.py    # basic ingestion + query test
  ├─ notebooks/
  │   └─ demo.ipynb       # example workflow
  ├─ requirements.txt
  ├─ README.md
  ├─ LICENSE
  └─ .gitignore
```

---

## Usage

### 1. Add your notes
Put `.md`, `.txt`, or `.pdf` files into the `notes/` directory.

### 2. Build the index
```bash
python -m src.build_index --notes_dir notes --index_dir artifacts/index
```

### 3. Ask a question
```bash
python -m src.run --question "What are the course tips?" --index_dir artifacts/index --k 3
```

---

## Example Run

**Input:**
```bash
python -m src.run --question "What are the course tips?" --index_dir artifacts/index --k 3
```

**Output:**
```
Q: What are the course tips?

Answer (draft):
Course Tips
- Always read the syllabus carefully.
- Attend office hours at least once early in the semester.
- Form a small study group of 2–3 people.
- Start assignments early; even 30 minutes can save stress later.
- Review past exams if available.

Sources:
[1] notes\course_tips.txt#chunk-0 (score=0.670)
[2] notes\faq.md#chunk-0 (score=0.101)
[3] notes\example.md#chunk-0 (score=0.099)
```

---

##  Development
Run tests:
```bash
pytest -q
```

Start the API server (optional):
```bash
uvicorn src.api:app --reload
```

Query via API:
```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" ^
  -d "{\"question\":\"What are the course tips?\",\"k\":3}"
```

---



