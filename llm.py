"""llm.py — one shared OpenAI model client for all agents."""
import os

from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()


def get_model_client() -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,  # same complaint -> same answer (more predictable)
    )