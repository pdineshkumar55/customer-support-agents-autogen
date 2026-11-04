"""Log Agent — saves every ticket to logs/tickets.csv and decides escalation."""
import csv
import threading
from datetime import datetime
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool

from guardrails import (SAFE_FALLBACK_REPLY, SELF_HARM_REPLY, injection_detected, moderate,
                        normalise_field, unverified_numbers, valid_kb_sources)

LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "tickets.csv"
_log_lock = threading.Lock()  # one writer at a time, so ticket IDs never repeat
COLUMNS = [
    "ticket_id", "timestamp", "customer_name", "order_id", "complaint",
    "category", "urgency", "sentiment", "language", "flags", "kb_sources",
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
    if "prompt_injection" in flag_text:
        reasons.append("Possible prompt injection")
    if "unverified_claim" in flag_text:
        reasons.append("Unverified claim in reply")
    if "harmful_content" in flag_text:
        reasons.append("Harmful content")
    if "unsafe_reply" in flag_text:
        reasons.append("Unsafe reply replaced")
    if "safety_check_unavailable" in flag_text:
        reasons.append("Safety check unavailable")
    if "invalid_field" in flag_text:
        reasons.append("Invalid agent output")
    if confidence.strip().lower() == "low":
        reasons.append("Low confidence")

    if reasons:
        return "Escalated", " + ".join(reasons)
    return "Resolved", ""


def _upgrade_log_columns() -> None:
    """If the log was created before a new column was added, rewrite it with the new header."""
    if not LOG_FILE.exists():
        return
    with LOG_FILE.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == COLUMNS:
            return
        rows = list(reader)
    with LOG_FILE.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
        writer.writeheader()
        for row in rows:
            # old tickets had no language column; they were all English
            writer.writerow({c: row.get(c) or ("English" if c == "language" else "")
                             for c in COLUMNS})


def _next_ticket_id() -> str:
    """TKT-0001, TKT-0002, ... based on how many rows are already in the log."""
    if not LOG_FILE.exists():
        return "TKT-0001"
    with LOG_FILE.open(encoding="utf-8-sig", newline="") as f:
        count = sum(1 for _ in csv.DictReader(f))
    return f"TKT-{count + 1:04d}"


def _add_flag(flags: str, flag: str) -> str:
    """Append a flag to the comma-separated flags ('none' is replaced)."""
    if flag.split("(")[0] in flags:
        return flags
    return flag if flags.strip().lower() in ("", "none") else f"{flags}, {flag}"


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
    language: str = "English",
    customer_name: str = "N/A",
    order_id: str = "N/A",
) -> str:
    """Save one support ticket to the log file and decide if it must be escalated
    to human support. Returns the ticket ID, status and escalation reason."""
    # ---------- guardrails: checked in Python, so the LLM can't skip them ----------
    flags = flags.strip() or "none"

    # the agents' answers must be allowed values ("high." is fixed to "High")
    fields = {"category": category, "urgency": urgency,
              "sentiment": sentiment, "confidence": confidence}
    invalid = []
    for name, value in fields.items():
        fields[name], ok = normalise_field(name, value)
        if not ok:
            invalid.append(name)
    category, urgency, sentiment, confidence = fields.values()
    if not valid_kb_sources(kb_sources):
        invalid.append("kb_sources")
    if not response.strip():
        invalid.append("response")
    if invalid:
        flags = _add_flag(flags, f"invalid_field({', '.join(invalid)})")

    if injection_detected(f"{customer_name}\n{order_id}\n{complaint}"):
        flags = _add_flag(flags, "prompt_injection")

    unverified = unverified_numbers(response, kb_sources, complaint, order_id, customer_name)
    if unverified:
        flags = _add_flag(flags, f"unverified_claim({', '.join(unverified)})")

    checked = moderate(complaint, response or "-")
    if checked is None:  # API down: let a human check it
        flags = _add_flag(flags, "safety_check_unavailable")
    else:
        harmful, reply_unsafe = checked
        if reply_unsafe:
            flags = _add_flag(flags, "unsafe_reply")
            response = SAFE_FALLBACK_REPLY
        if harmful:
            flags = _add_flag(flags, f"harmful_content({', '.join(sorted(harmful))})")
            if any(c.startswith("self_harm") for c in harmful):
                response = SELF_HARM_REPLY

    status, reason = decide_escalation(urgency, sentiment, flags, kb_sources, confidence)
    row = {
        "ticket_id": "",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "customer_name": customer_name or "N/A",
        "order_id": order_id or "N/A",
        "complaint": complaint,
        "category": category,
        "urgency": urgency,
        "sentiment": sentiment,
        "language": language,
        "flags": flags,
        "kb_sources": kb_sources,
        "response": response,
        "confidence": confidence,
        "status": status,
        "escalation_reason": reason,
    }

    with _log_lock:
        _upgrade_log_columns()
        row["ticket_id"] = _next_ticket_id()
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

1. Call the log_ticket tool ONCE. Copy the values exactly from the conversation
   (the complaint, customer name and order ID are filled in automatically):
   - category, urgency, sentiment, flags, language: from the query_agent
   - kb_sources: from the knowledge_agent's KB_SOURCES line (NONE if none)
   - response: the full text under REPLY from the resolution_agent
   - confidence: from the resolution_agent's CONFIDENCE line
2. After the tool returns, reply in EXACTLY this format:

TICKET: <ticket_id>
STATUS: <Resolved or Escalated>
ESCALATION_REASON: <reason, or none>
LOGGED"""


def create_log_agent(model_client, complaint: str, customer_name: str = "",
                     order_id: str = "") -> AssistantAgent:
    def save_ticket(category: str, urgency: str, sentiment: str, flags: str, kb_sources: str,
                    response: str, confidence: str, language: str = "English") -> str:
        # complaint, name and order ID come from the real ticket, never from the LLM
        return log_ticket(complaint, category, urgency, sentiment, flags, kb_sources,
                          response, confidence, language,
                          customer_name.strip() or "N/A", order_id.strip() or "N/A")

    return AssistantAgent(
        name="log_agent",
        description="Records the ticket to the log file and decides whether to escalate to a human.",
        model_client=model_client,
        tools=[FunctionTool(save_ticket, name="log_ticket", description=log_ticket.__doc__)],
        reflect_on_tool_use=True,
        system_message=SYSTEM_MESSAGE,
    )