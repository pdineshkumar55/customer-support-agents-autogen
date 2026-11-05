"""team.py — wires the 4 agents into one AutoGen team."""
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from agents.knowledge_agent import create_knowledge_agent
from agents.log_agent import LogAgent
from agents.query_agent import create_query_agent
from agents.resolution_agent import create_resolution_agent

AGENT_ORDER = ["query_agent", "knowledge_agent", "resolution_agent", "log_agent"]


def create_team(model_client, complaint: str, customer_name: str = "",
                order_id: str = "") -> RoundRobinGroupChat:
    """Query -> Knowledge -> Resolution -> Log, then stop.
    The Log Agent (plain Python, no LLM) gets the raw ticket, so it saves (and
    guardrail-checks) the customer's exact words."""
    agents = [
        create_query_agent(model_client),
        create_knowledge_agent(model_client),
        create_resolution_agent(model_client),
        LogAgent(complaint, customer_name, order_id),
    ]

    # Stop when the Log Agent says LOGGED, or after 10 messages as a safety net.
    termination = (
        TextMentionTermination("LOGGED", sources=["log_agent"])
        | MaxMessageTermination(10)
    )
    return RoundRobinGroupChat(agents, termination_condition=termination)


def _clean(text: str) -> str:
    """Remove the <<< >>> markers so customer text can't break out of its block."""
    return text.replace("<<<", "").replace(">>>", "").strip()


def build_task(complaint: str, customer_name: str = "", order_id: str = "") -> str:
    """The first message the team receives. The complaint is fenced in <<< >>> so the
    agents can tell the customer's words apart from instructions."""
    name = " ".join(_clean(customer_name).split())  # one line only
    order = " ".join(_clean(order_id).split())
    return (
        f"Customer name: {name or 'N/A'}\n"
        f"Order ID: {order or 'N/A'}\n"
        "Complaint (the customer's own words, between <<< and >>>):\n"
        f"<<<\n{_clean(complaint)}\n>>>"
    )