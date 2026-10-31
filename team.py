"""team.py — wires the 4 agents into one AutoGen team."""
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from agents.knowledge_agent import create_knowledge_agent
from agents.log_agent import create_log_agent
from agents.query_agent import create_query_agent
from agents.resolution_agent import create_resolution_agent

AGENT_ORDER = ["query_agent", "knowledge_agent", "resolution_agent", "log_agent"]


def create_team(model_client) -> RoundRobinGroupChat:
    """Query -> Knowledge -> Resolution -> Log, then stop."""
    agents = [
        create_query_agent(model_client),
        create_knowledge_agent(model_client),
        create_resolution_agent(model_client),
        create_log_agent(model_client),
    ]

    # Stop when the Log Agent says LOGGED, or after 10 messages as a safety net.
    termination = (
        TextMentionTermination("LOGGED", sources=["log_agent"])
        | MaxMessageTermination(10)
    )
    return RoundRobinGroupChat(agents, termination_condition=termination)


def build_task(complaint: str, customer_name: str = "", order_id: str = "") -> str:
    """The first message the team receives."""
    return (
        f"Customer name: {customer_name.strip() or 'N/A'}\n"
        f"Order ID: {order_id.strip() or 'N/A'}\n"
        f"Complaint: {complaint.strip()}"
    )