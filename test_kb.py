"""Retrieval: vector (FAISS) + keyword (BM25) search, fused.

The BM25 tests run offline; the rest call the OpenAI embeddings API (one small call per query).
Run: pytest test_kb.py
"""
import pytest

from kb_store import BM25, MIN_SCORE, search_kb, search_kb_raw, tokenize
from support_cases import CASE_PARAMS

needs_openai = pytest.mark.needs_openai


# ---------- BM25 on its own (no API) ----------
DOCS = [
    "Is Cash on Delivery (COD) available? COD is available for orders up to 20,000.",
    "Track your order: the courier name and the AWB (tracking) number are shown.",
    "Refunds to a bank account are sent by NEFT within 5 working days.",
]


def test_tokenize_keeps_ids_whole_and_drops_stopwords():
    assert tokenize("Is my order ORD-4512 a 6-digit OTP?") == ["order", "ord-4512", "6-digit", "otp"]


def test_bm25_ranks_the_chunk_with_the_term_first():
    bm25 = BM25(DOCS)
    for term, best in [("cod", 0), ("awb", 1), ("neft", 2)]:
        scores = bm25.scores([term])
        assert scores.index(max(scores)) == best
        assert sorted(scores)[1] == 0  # the other chunks don't contain it


@pytest.mark.parametrize("query, expected", [
    ("Can I pay COD?", {"cod"}),
    ("can i pay cod", {"cod"}),                        # acronym from the documents, any case
    ("Where is the AWB for ORD-4512?", {"awb", "ord-4512"}),
    ("I am waiting since 9 AM, what is my order ID?", set()),  # short acronyms don't count
    ("Gift cards for a company of 500 employees", set()),  # plain words and plain numbers don't
])
def test_exact_terms(query, expected):
    assert BM25(DOCS).exact_terms(query) == expected


# ---------- hybrid search over the real knowledge base ----------
@needs_openai
@pytest.mark.parametrize("case", CASE_PARAMS)
def test_best_match(case):
    best = search_kb_raw(case.complaint)[0]
    if case.top_source == "NONE":
        assert best["score"] < MIN_SCORE and not best["exact_terms"], best["source"]
        assert search_kb(case.complaint).startswith("NO_MATCH")
    else:
        assert best["source"] == case.top_source
        assert best["score"] >= MIN_SCORE
        assert f"[Source: {case.top_source}" in search_kb(case.complaint)


@needs_openai
@pytest.mark.parametrize("query, term, source, must_contain", [
    ("Can I pay COD?", "cod", "product_faq.pdf", "COD"),
    ("is cod available for my order ORD-4512", "cod", "product_faq.pdf", "COD"),
    ("Where do I find the AWB number?", "awb", "shipping_policy.pdf", "AWB"),
    ("How long does a NEFT refund take?", "neft", "refund_policy.pdf", "NEFT"),
    ("Do you offer EMI?", "emi", "product_faq.pdf", "EMI"),
])
def test_exact_term_is_found(query, term, source, must_contain):
    best = search_kb_raw(query)[0]
    assert best["source"] == source
    assert must_contain in best["text"]
    assert term in best["exact_terms"]
    assert f"exact match: {term.upper()}" in search_kb(query)
