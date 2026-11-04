"""Step 5 test: run complaints through the full 4-agent team."""
import asyncio
import sys

import pandas as pd
from autogen_agentchat.ui import Console

from agents.log_agent import LOG_FILE
from llm import get_model_client
from team import build_task, create_team

TEST_CASES = [
    ("Priya", "ORD-4512", "Where is my order? It was supposed to arrive yesterday."),
    ("Arjun", "ORD-3301", "I want to return the shoes I bought last week. How do I get a refund?"),
    ("Meena", "", "I forgot my password and can't log in."),
    ("Karthik", "ORD-2290", "The blender I received is broken. Can I get a replacement?"),
    ("Ravi", "ORD-1187", "You charged my card three times! Fix this now or I'm contacting my bank and a lawyer."),
    ("Divya", "", "Do you sell gift cards for a company of 500 employees with custom branding?"),
]


async def run_one(client, name, order_id, complaint):
    team = create_team(client, complaint, name, order_id)  # fresh team = fresh memory for every ticket
    result = await Console(team.run_stream(task=build_task(complaint, name, order_id)))
    print(f"\n>>> Stopped because: {result.stop_reason}\n" + "#" * 80)


async def main():
    client = get_model_client()
    # python test_team.py 3   -> runs only test case 3
    cases = [TEST_CASES[int(sys.argv[1]) - 1]] if len(sys.argv) > 1 else TEST_CASES
    for case in cases:
        await run_one(client, *case)
    await client.close()

    pd.set_option("display.max_columns", None, "display.width", 200)
    print(pd.read_csv(LOG_FILE)[["ticket_id", "customer_name", "category", "urgency",
                                 "sentiment", "kb_sources", "confidence", "status",
                                 "escalation_reason"]].fillna(""))


if __name__ == "__main__":
    asyncio.run(main())