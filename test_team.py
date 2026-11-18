"""End to end: each complaint through the full 4-agent team (real LLM calls).

Checks the logged category and status. Urgency and sentiment are not asserted,
because the model may reasonably vary there (e.g. Low vs Medium).
Run: pytest test_team.py            (all 8, ~16 LLM calls)
     pytest test_team.py -k charged (only case 5)
     pytest -m "not llm"            (everything except this file)
"""
import asyncio

import pandas as pd
import pytest
from autogen_agentchat.messages import TextMessage

from llm import get_model_client
from support_cases import CASE_PARAMS
from team import build_task, create_team

pytestmark = [pytest.mark.needs_openai, pytest.mark.llm]


async def _run(case):
    client = get_model_client()
    try:
        team = create_team(client, case.complaint, case.customer_name, case.order_id)
        return await team.run(task=build_task(case.complaint, case.customer_name, case.order_id))
    finally:
        await client.close()


@pytest.mark.parametrize("case", CASE_PARAMS)
def test_team_handles_complaint(case, temp_log):
    result = asyncio.run(_run(case))
    last = result.messages[-1]
    assert isinstance(last, TextMessage) and last.source == "log_agent", result.stop_reason
    assert "LOGGED" in last.content


    row = pd.read_csv(temp_log, encoding="utf-8-sig", keep_default_na=False).iloc[-1]
    details = f"{row['category']} / {row['status']} ({row['escalation_reason']}) flags={row['flags']}"
    assert row["category"] == case.category, details
    assert row["status"] == case.status, details
    assert row["complaint"] == case.complaint
