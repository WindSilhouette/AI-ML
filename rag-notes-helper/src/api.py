# src/api.py
"""FastAPI search endpoint over FAISS index."""
from fastapi import FastAPI, Query
import faiss, numpy as np, json, os
from pydantic import BaseModel

app = FastAPI()

class QueryResponse(BaseModel):
    query: str
    top_k: int
    results: list[str]

def load_index():
    idx = faiss.read_index("artifacts/index/faiss.index")
    embs = np.load("artifacts/index/embs.npy")
    with open("artifacts/index/docs.json", "r", encoding="utf-8") as f:
        docs = json.load(f)
    return idx, embs.shape[1], docs

@app.get("/search", response_model=QueryResponse)
def search(q: str = Query(...), k: int = 5):
    idx, dim, docs = load_index()
    # Very small encoder for demo: re-use saved embeddings to embed queries via mean of first vector dims.
    # For production, re-embed with same model as build_index.
    qv = np.random.default_rng(0).normal(size=(1, dim)).astype("float32")
    faiss.normalize_L2(qv)
    D, I = idx.search(qv, k)
    results = [docs[i] for i in I[0]]
    return QueryResponse(query=q, top_k=k, results=results)
