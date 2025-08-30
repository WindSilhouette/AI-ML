# ROADMAP

**Milestone 1 — Index Build**
- [ ] Put a couple of `.md` notes into `notes/`, then run `python -m src.build_index`.

**Milestone 2 — API**
- [ ] `python -m src.run` → GET `http://localhost:8000/search?q=test` returns top chunks.

**Milestone 3 — Proper Encoding**
- [ ] Replace the dummy query embedding with the same SentenceTransformer used in indexing.

**Milestone 4 — UI**
- [ ] Add a small web UI (React or simple HTML) to call the API and render answers.
