# tests/test_smoke.py
from pathlib import Path
import subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes"
INDEX = ROOT / "artifacts" / "index"

def test_build_and_query():
    # build
    subprocess.check_call([sys.executable, "-m", "src.build_index",
                           "--notes_dir", str(NOTES),
                           "--index_dir", str(INDEX)])
    assert (INDEX / "faiss.index").exists()
    # query
    out = subprocess.check_output([sys.executable, "-m", "src.run",
                                   "--question", "example",
                                   "--index_dir", str(INDEX),
                                   "--k", "3"], text=True)
    assert "Q:" in out and "notes/example.md" in out
