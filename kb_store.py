"""
kb_store.py — the company knowledge base (simple RAG).

1. Reads every PDF in knowledge_base/
2. Splits the text into small overlapping chunks
3. Turns each chunk into an embedding (OpenAI) and stores it in a FAISS index
4. search_kb(query) returns the best-matching passages with their file names

The index is saved in faiss_index/ and rebuilt automatically
whenever a PDF is added, removed or changed.
"""

import json
import os
import re
import threading
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge_base"
INDEX_DIR = BASE_DIR / "faiss_index"
INDEX_FILE = INDEX_DIR / "index.faiss"
CHUNKS_FILE = INDEX_DIR / "chunks.json"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks
MIN_SCORE = float(os.getenv("KB_MIN_SCORE", "0.30"))  # below this = "no match"

client = OpenAI()  # reads OPENAI_API_KEY from the environment


# ---------- 1. Read PDFs ----------
def read_pdf(path: Path) -> str:
    """Extract text from a PDF and tidy up the lines."""
    reader = PdfReader(str(path))
    raw = "\n".join(page.extract_text() or "" for page in reader.pages)

    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or re.fullmatch(r"Page \d+", line):
            continue                          # drop blank lines and page footers
        if line == "•":
            lines.append("- ")                # bullet symbol sits on its own line
        elif re.fullmatch(r"\d{1,2}", line):
            lines.append(f"{line}. ")         # step number sits on its own line
        elif lines and re.fullmatch(r"(- |\d{1,2}\. )", lines[-1]):
            lines[-1] += line                 # join bullet / number with its text
        else:
            lines.append(line)
    return "\n".join(lines)


# ---------- 2. Chunk ----------
def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks, cutting at line breaks where possible."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            cut = text.rfind("\n", start + CHUNK_SIZE // 2, end)
            if cut != -1:
                end = cut
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        # start the next chunk at the beginning of a line inside the overlap
        line_start = text.find("\n", end - CHUNK_OVERLAP, end)
        start = line_start + 1 if line_start != -1 else end
    return chunks


# ---------- 3. Embed ----------
def embed(texts: list[str]) -> np.ndarray:
    """Get embeddings from OpenAI and normalise them (so inner product = cosine similarity)."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors = np.array([item.embedding for item in response.data], dtype="float32")
    faiss.normalize_L2(vectors)
    return vectors


# ---------- 4. Build / load the index ----------
def _kb_fingerprint() -> dict:
    """File name -> last-modified time. If this changes, the index is rebuilt."""
    return {p.name: p.stat().st_mtime for p in sorted(KB_DIR.glob("*.pdf"))}


def build_index():
    """Read all PDFs, chunk, embed and save the FAISS index to disk."""
    records = []
    for pdf in sorted(KB_DIR.glob("*.pdf")):
        for chunk in chunk_text(read_pdf(pdf)):
            records.append({"source": pdf.name, "text": chunk})

    if not records:
        raise RuntimeError(f"No PDF text found in {KB_DIR}/")

    vectors = embed([r["text"] for r in records])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    CHUNKS_FILE.write_text(
        json.dumps({"fingerprint": _kb_fingerprint(), "chunks": records}, indent=2),
        encoding="utf-8",
    )
    print(f"Built FAISS index: {len(records)} chunks from {len(_kb_fingerprint())} PDFs")
    return index, records


def load_index():
    """Load the saved index, or rebuild it if the PDFs changed."""
    if INDEX_FILE.exists() and CHUNKS_FILE.exists():
        saved = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
        if saved["fingerprint"] == _kb_fingerprint():
            return faiss.read_index(str(INDEX_FILE)), saved["chunks"]
    return build_index()


_index, _chunks = None, None
_index_lock = threading.Lock()  # Streamlit sessions are threads: build the index only once


def _get_index():
    global _index, _chunks
    with _index_lock:
        if _index is None:
            _index, _chunks = load_index()
    return _index, _chunks


# ---------- 5. The tool the Knowledge Agent will call ----------
def search_kb_raw(query: str, top_k: int = 3) -> list[dict]:
    """Return the top matches as dicts: source, score, text (used for testing)."""
    index, chunks = _get_index()
    scores, ids = index.search(embed([query]), top_k)
    return [
        {"source": chunks[i]["source"], "score": round(float(s), 3), "text": chunks[i]["text"]}
        for s, i in zip(scores[0], ids[0])
        if i != -1
    ]


def search_kb(query: str) -> str:
    """Search the ShopNest policy documents (shipping, refunds, account/login, product FAQ)
    and return the most relevant passages with their source file names.
    Returns 'NO_MATCH' if nothing relevant is found."""
    results = [r for r in search_kb_raw(query) if r["score"] >= MIN_SCORE]
    if not results:
        return "NO_MATCH: no relevant passage found in the knowledge base."
    return "\n\n".join(
        f"[Source: {r['source']} | score {r['score']}]\n{r['text']}" for r in results
    )