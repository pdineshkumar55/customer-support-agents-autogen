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


def test_no_pii_reaches_the_agents_or_the_log(temp_log):
    complaint = ("You charged my card 4111 1111 1111 1111 twice for order ORD-1187. "
                 "Call me on 9876543210 or email ravi.k@gmail.com")
    case = type("C", (), dict(complaint=complaint, customer_name="Ravi", order_id="ORD-1187"))
    result = asyncio.run(_run(case))
    everything = "\n".join(m.to_text() for m in result.messages) + temp_log.read_text(encoding="utf-8-sig")
    for secret in ("4111 1111 1111 1111", "9876543210", "ravi.k@gmail.com"):
        assert secret not in everything
    assert "[CARD ending 1111]" in everything
