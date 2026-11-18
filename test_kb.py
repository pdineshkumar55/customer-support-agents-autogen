"""Retrieval: each complaint's best FAISS match is the right policy PDF.

Calls the OpenAI embeddings API (one small call per query).
Run: pytest test_kb.py
"""
import pytest

from kb_store import MIN_SCORE, search_kb, search_kb_raw
from support_cases import CASE_PARAMS

pytestmark = pytest.mark.needs_openai


@pytest.mark.parametrize("case", CASE_PARAMS)
def test_best_match(case):
    results = search_kb_raw(case.complaint)
    best = results[0]
    if case.top_source == "NONE":
        assert best["score"] < MIN_SCORE, f"unexpected match: {best['source']} {best['score']}"
        assert search_kb(case.complaint).startswith("NO_MATCH")
    else:
        assert best["source"] == case.top_source
        assert best["score"] >= MIN_SCORE
        assert f"[Source: {case.top_source}" in search_kb(case.complaint)
