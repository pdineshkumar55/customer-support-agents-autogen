"""Shared pytest setup.

- Loads .env. Without OPENAI_API_KEY a placeholder key is set so the modules still import,
  and every test marked `needs_openai` is skipped.
- `temp_log` points the Log Agent at a throwaway CSV, so logs/tickets.csv is never touched.
"""
import os

import pytest
from dotenv import load_dotenv

load_dotenv()
HAS_OPENAI_KEY = bool(os.getenv("OPENAI_API_KEY"))
if not HAS_OPENAI_KEY:
    os.environ["OPENAI_API_KEY"] = "sk-test-placeholder"  # kb_store creates OpenAI() on import


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="OPENAI_API_KEY not set")
    for item in items:
        if "needs_openai" in item.keywords and not HAS_OPENAI_KEY:
            item.add_marker(skip)


@pytest.fixture
def temp_log(tmp_path, monkeypatch):
    """Write tickets to a temp file; returns its path."""
    from agents import log_agent
    path = tmp_path / "tickets.csv"
    monkeypatch.setattr(log_agent, "LOG_FILE", path)
    return path
