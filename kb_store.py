"""
kb_store.py — the company knowledge base (simple RAG).

1. Reads every PDF in knowledge_base/
2. Splits the text into small overlapping chunks
3. Turns each chunk into an embedding (OpenAI) and stores it in a FAISS index
4. Builds a BM25 keyword index over the same chunks (exact terms such as "COD", "OTP")
5. search_kb(query) fuses both rankings and returns the best passages with their file names

The index is saved in faiss_index/ and rebuilt automatically
whenever a PDF is added, removed or changed.
"""

import json
import math
import os
import re
import threading
from collections import Counter
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
MIN_SCORE = float(os.getenv("KB_MIN_SCORE", "0.30"))  # vector score below this = "no match"
RRF_K = 60            # reciprocal rank fusion constant (the usual default)
BM25_K1, BM25_B = 1.5, 0.75

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


def source_text(source: str) -> str:
    """All indexed text of one PDF ('' if it is not in the knowledge base). Used by guardrails.py."""
    _, chunks = _get_index()
    return "\n".join(c["text"] for c in chunks if c["source"] == source)


def known_sources() -> set[str]:
    """File names of every PDF in the knowledge base. Used by guardrails.py."""
    _, chunks = _get_index()
    return {c["source"] for c in chunks}


# ---------- 5. BM25 keyword index ----------
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")  # keeps "ord-4512" and "6-digit" whole
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")
STOPWORDS = set("""a am an and are as at be by can do does for from get got had has have how i if in
into is it its me my no not of on or our so than that the their them then there these they this
pm to too up us was we were what when where which who why will with you your""".split())


def tokenize(text: str) -> list[str]:
    return [w for w in _TOKEN_RE.findall(text.lower()) if w not in STOPWORDS]


class BM25:
    """Okapi BM25 over the chunks (19 chunks: plain Python is plenty)."""

    def __init__(self, texts: list[str]):
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / len(self.docs)
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}
        # acronyms written in the documents (COD, OTP, AWB, ...): exact terms worth matching
        self.acronyms = {a.lower() for t in texts for a in _ACRONYM_RE.findall(t)}

    def scores(self, query_terms: list[str]) -> list[float]:
        out = []
        for doc, length in zip(self.docs, self.lengths):
            s = 0.0
            for term in query_terms:
                tf = doc.get(term, 0)
                if tf:
                    norm = tf + BM25_K1 * (1 - BM25_B + BM25_B * length / self.avg_len)
                    s += self.idf[term] * tf * (BM25_K1 + 1) / norm
            out.append(s)
        return out

    def exact_terms(self, query: str) -> set[str]:
        """Query terms that are identifiers, not ordinary words:
        - written in capitals (COD, AWB), or an acronym the documents use, also when typed
          lowercase ("cod"); at least 3 letters, so "ID" or "AM" do not count
        - letters mixed with digits (ORD-4512, 6-digit)
        A chunk containing one always counts as a match, even when its meaning-based
        score is low."""
        caps = {a.lower() for a in _ACRONYM_RE.findall(query)}
        terms = set()
        for w in tokenize(query):
            acronym = len(w) >= 3 and (w in caps or w in self.acronyms)
            identifier = bool(re.search(r"\d", w) and re.search(r"[a-z]", w))
            if acronym or identifier:
                terms.add(w)
        return terms


_bm25: BM25 | None = None
_bm25_chunks = None


def _get_bm25(chunks: list[dict]) -> BM25:
    global _bm25, _bm25_chunks
    with _index_lock:
        if _bm25_chunks is not chunks:  # built once per loaded index
            _bm25, _bm25_chunks = BM25([c["text"] for c in chunks]), chunks
    return _bm25


# ---------- 6. Hybrid search: the tool the Knowledge Agent will call ----------
def search_kb_raw(query: str, top_k: int | None = 3) -> list[dict]:
    """Top matches (all if top_k is None) as dicts, best first (used by search_kb and tests):
    source, text, score (vector cosine), keyword (BM25), exact_terms, fused (RRF)."""
    index, chunks = _get_index()
    bm25 = _get_bm25(chunks)

    # vector pass: score every chunk (the knowledge base is small)
    cosine, ids = index.search(embed([query]), len(chunks))
    vector = dict(zip(ids[0].tolist(), cosine[0].tolist()))
    # keyword pass
    keyword = bm25.scores(tokenize(query))
    exact = bm25.exact_terms(query)

    # reciprocal rank fusion: rank-based, so cosine and BM25 scales never need matching
    fused = {i: 1 / (RRF_K + rank) for rank, i in enumerate(sorted(vector, key=vector.get, reverse=True), 1)}
    for rank, i in enumerate(sorted((i for i in range(len(chunks)) if keyword[i] > 0),
                                    key=lambda i: keyword[i], reverse=True), 1):
        fused[i] += 1 / (RRF_K + rank)

    results = []
    for i in sorted(fused, key=fused.get, reverse=True)[:top_k]:
        words = set(tokenize(chunks[i]["text"]))
        results.append({
            "source": chunks[i]["source"],
            "score": round(vector[i], 3),
            "keyword": round(keyword[i], 3),
            "exact_terms": sorted(exact & words),
            "fused": round(fused[i], 4),
            "text": chunks[i]["text"],
        })
    return results


def _relevant(r: dict) -> bool:
    """Close in meaning, or contains an exact identifier / acronym from the query."""
    return r["score"] >= MIN_SCORE or bool(r["exact_terms"])


def search_kb(query: str) -> str:
    """Search the ShopNest policy documents (shipping, refunds, account/login, product FAQ)
    and return the most relevant passages with their source file names.
    Returns 'NO_MATCH' if nothing relevant is found."""
    # gate first, then keep the best 3, so a keyword-only chunk never pushes out a relevant one
    results = [r for r in search_kb_raw(query, top_k=None) if _relevant(r)][:3]
    if not results:
        return "NO_MATCH: no relevant passage found in the knowledge base."
    out = []
    for r in results:
        exact = f" | exact match: {', '.join(t.upper() for t in r['exact_terms'])}" if r["exact_terms"] else ""
        out.append(f"[Source: {r['source']} | score {r['score']}{exact}]\n{r['text']}")
    return "\n\n".join(out)
