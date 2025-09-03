import argparse, json
from pathlib import Path
import faiss, numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_index(index_dir: Path):
    index = faiss.read_index(str(index_dir / "faiss.index"))
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    metas  = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    return index, chunks, metas

def synthesize_answer(question: str, hits: list, max_lines: int = 6) -> str:
    # Simple extractive draft from the top contexts (no API needed)
    lines = []
    for h in hits:
        for ln in h["text"].splitlines():
            ln = ln.strip()
            if ln and ln not in lines:
                lines.append(ln)
            if len(lines) >= max_lines:
                break
        if len(lines) >= max_lines:
            break
    return "\n".join(lines) if lines else "(No content found in sources.)"

def ask(question: str, index_dir: Path, k: int = 5):
    model = SentenceTransformer(EMBED_MODEL)
    qv = model.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(qv)

    index, chunks, metas = load_index(index_dir)

    # cap k to number of vectors to avoid repeats
    k = min(k, index.ntotal) if hasattr(index, "ntotal") else k
    if k <= 0:
        return []

    D, I = index.search(qv, k)
    seen = set()
    results = []
    for rank, idx in enumerate(I[0].tolist()):
        if idx in seen:
            continue
        seen.add(idx)
        results.append({
            "rank": len(results) + 1,
            "score": float(D[0][rank]),
            "text": chunks[idx],
            "meta": metas[idx]
        })
    return results

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--index_dir", default="artifacts/index")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    hits = ask(args.question, index_dir, args.k)

    print(f"Q: {args.question}\n")
    if not hits:
        print("No results. Try rebuilding the index or adding notes.")
    else:
        # Draft answer from retrieved context
        answer = synthesize_answer(args.question, hits)
        print("Answer (draft):")
        print(answer, "\n")
        print("Sources:")
        for h in hits:
            path = h["meta"]["path"].replace("/", "\\")
            cid = h["meta"]["chunk_id"]
            print(f"[{h['rank']}] {path}#chunk-{cid} (score={h['score']:.3f})")
