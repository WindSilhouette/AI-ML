# src/build_index.py
"""Build a FAISS index from Markdown/PDF notes.
Logic: Chunk text -> embed -> store vectors + metadata.
"""
import os, glob
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def read_texts(notes_dir: str) -> List[str]:
    texts = []
    for path in glob.glob(os.path.join(notes_dir, "**/*.md"), recursive=True):
        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts

def chunk_text(text: str, size=400, overlap=40):
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+size]
        chunks.append(" ".join(chunk))
        i += size - overlap
    return chunks

def build(notes_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    docs = []
    for t in read_texts(notes_dir):
        docs.extend(chunk_text(t))
    if not docs:
        raise SystemExit("No documents found.")
    embs = model.encode(docs, convert_to_numpy=True, show_progress_bar=False)
    index = faiss.IndexFlatIP(embs.shape[1])
    faiss.normalize_L2(embs)
    index.add(embs)
    np.save(os.path.join(out_dir, "embs.npy"), embs)
    with open(os.path.join(out_dir, "docs.json"), "w", encoding="utf-8") as f:
        import json; json.dump(docs, f)
    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))

if __name__ == "__main__":
    build("notes", "artifacts/index")
