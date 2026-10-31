"""Log Agent — saves every ticket to logs/tickets.csv and decides escalation."""
import csv
from datetime import datetime
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent

LOG_FILE = Path("logs") / "tickets.csv"
COLUMNS = [
    "ticket_id", "timestamp", "customer_name", "order_id", "complaint",
    "category", "urgency", "sentiment", "flags", "kb_sources",
    "response", "confidence", "status", "escalation_reason",
]


# ---------- Escalation rule (plain Python, so it is always applied the same way) ----------
def decide_escalation(urgency: str, sentiment: str, flags: str,
                      kb_sources: str, confidence: str) -> tuple[str, str]:
    """Return (status, reason). Escalate if ANY rule is true."""
    reasons = []
    if urgency.strip().lower() == "high":
        reasons.append("High urgency")
    if sentiment.strip().lower() == "angry":
        reasons.append("Angry customer")
    if kb_sources.strip().upper() in ("", "NONE"):
        reasons.append("No matching document")
    flag_text = flags.lower()
    if "legal_threat" in flag_text:
        reasons.append("Legal threat")
    if "fraud" in flag_text:
        reasons.append("Possible fraud")
    if "payment_dispute" in flag_text:
        reasons.append("Payment dispute")
    if confidence.strip().lower() == "low":
        reasons.append("Low confidence")

    if reasons:
        return "Escalated", " + ".join(reasons)
    return "Resolved", ""


def _next_ticket_id() -> str:
    """TKT-0001, TKT-0002, ... based on how many rows are already in the log."""
    if not LOG_FILE.exists():
        return "TKT-0001"
    with LOG_FILE.open(encoding="utf-8-sig", newline="") as f:
        count = sum(1 for _ in csv.DictReader(f))
    return f"TKT-{count + 1:04d}"


# ---------- The tool the Log Agent calls ----------
def log_ticket(
    complaint: str,
    category: str,
    urgency: str,
    sentiment: str,
    flags: str,
    kb_sources: str,
    response: str,
    confidence: str,
    customer_name: str = "N/A",
    order_id: str = "N/A",
) -> str:
    """Save one support ticket to the log file and decide if it must be escalated
    to human support. Returns the ticket ID, status and escalation reason."""
    status, reason = decide_escalation(urgency, sentiment, flags, kb_sources, confidence)
    row = {
        "ticket_id": _next_ticket_id(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "customer_name": customer_name or "N/A",
        "order_id": order_id or "N/A",
        "complaint": complaint,
        "category": category,
        "urgency": urgency,
        "sentiment": sentiment,
        "flags": flags,
        "kb_sources": kb_sources,
        "response": response,
        "confidence": confidence,
        "status": status,
        "escalation_reason": reason,
    }

    LOG_FILE.parent.mkdir(exist_ok=True)
    is_new = not LOG_FILE.exists()
    # utf-8-sig so Excel shows ₹ and other symbols correctly
    with LOG_FILE.open("a", encoding="utf-8-sig" if is_new else "utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)

    return (f"Ticket saved. ticket_id={row['ticket_id']}; status={status}; "
            f"escalation_reason={reason or 'none'}")


SYSTEM_MESSAGE = """You are the Log Agent for ShopNest customer support.
Your ONLY job is to record the ticket. Do not change or rewrite anything.

1. Call the log_ticket tool ONCE. Copy the values exactly from the conversation:
   - complaint, customer_name, order_id: from the original task (use "N/A" if missing)
   - category, urgency, sentiment, flags: from the query_agent
   - kb_sources: from the knowledge_agent's KB_SOURCES line (NONE if none)
   - response: the full text under REPLY from the resolution_agent
   - confidence: from the resolution_agent's CONFIDENCE line
2. After the tool returns, reply in EXACTLY this format:

TICKET: <ticket_id>
STATUS: <Resolved or Escalated>
ESCALATION_REASON: <reason, or none>
LOGGED"""


def create_log_agent(model_client) -> AssistantAgent:
    return AssistantAgent(
        name="log_agent",
        description="Records the ticket to the log file and decides whether to escalate to a human.",
        model_client=model_client,
        tools=[log_ticket],
        reflect_on_tool_use=True,
        system_message=SYSTEM_MESSAGE,
    )