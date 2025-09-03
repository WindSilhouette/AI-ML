import argparse, json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss, numpy as np
from pypdf import PdfReader
from pypdf.errors import PdfReadError, PdfStreamError

# Config
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100



def normalize_text(text: str) -> str:
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n"))

def read_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in [".txt", ".md"]:
        return normalize_text(path.read_text(encoding="utf-8", errors="ignore"))

    if ext == ".pdf":
        try:
            reader = PdfReader(str(path))
            raw = "\n".join((page.extract_text() or "") for page in reader.pages)
            return normalize_text(raw)
        except (PdfReadError, PdfStreamError, OSError, ValueError):
            # Not a valid PDF — skip by returning empty to ignore this file
            print(f"[warn] Skipping invalid PDF: {path}")
            return ""

    # Unknown extension => skip
    return ""



def chunk_text(text: str, chunk_chars: int = CHUNK_SIZE * 6, overlap_chars: int = CHUNK_OVERLAP * 6):

    if not text:
        return []
    chunks = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_chars, n)
        # try not to cut in the middle of a line: extend to next newline if close
        if end < n:
            nl = text.rfind("\n", start, end)  # prefer to break at the last newline before end
            if nl != -1 and end - nl < 120:    # if a newline is reasonably close, break there
                end = nl + 1
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap_chars)
    return chunks

def build_index(notes_dir: Path, index_dir: Path):
    model = SentenceTransformer(EMBED_MODEL)
    docs, metas, chunks = [], [], []

    for path in notes_dir.rglob("*"):
        if path.is_file():
            text = read_file(path)
            for i, chunk in enumerate(chunk_text(text)):
                chunks.append(chunk)
                metas.append({"path": str(path), "chunk_id": i})

    X = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    np.save(index_dir / "vecs.npy", X)
    (index_dir / "chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    (index_dir / "meta.json").write_text(json.dumps(metas, indent=2), encoding="utf-8")

    print(f"Built index with {len(chunks)} chunks from {len(metas)} docs.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes_dir", default="notes")
    ap.add_argument("--index_dir", default="artifacts/index")
    args = ap.parse_args()
    build_index(Path(args.notes_dir), Path(args.index_dir))
