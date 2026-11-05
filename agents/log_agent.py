"""Log Agent — saves every ticket to logs/tickets.csv and decides escalation."""
import asyncio
import csv
import re
import threading
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_core import CancellationToken

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
    """TKT-0001, TKT-0002, ... one more than the highest ID so far
    (not the row count, so a deleted row never causes a repeated ID)."""
    if not LOG_FILE.exists():
        return "TKT-0001"
    with LOG_FILE.open(encoding="utf-8-sig", newline="") as f:
        numbers = [int(m.group(1)) for row in csv.DictReader(f)
                   if (m := re.fullmatch(r"TKT-(\d+)", row.get("ticket_id") or ""))]
    return f"TKT-{max(numbers, default=0) + 1:04d}"


def _add_flag(flags: str, flag: str) -> str:
    """Append a flag to the comma-separated flags ('none' is replaced)."""
    if flag.split("(")[0] in flags:
        return flags
    return flag if flags.strip().lower() in ("", "none") else f"{flags}, {flag}"


# ---------- Save one ticket (called by the Log Agent) ----------
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
) -> dict:
    """Save one support ticket to the log file and decide if it must be escalated
    to human support. Returns the saved row (ticket_id, status, escalation_reason, ...)."""
    # ---------- guardrails: checked in Python, so they can't be skipped ----------
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

    return row


# ---------- Reading the other agents' answers ----------
def parse_reply(text: str) -> tuple[str, str]:
    """Split the Resolution Agent's message into (reply, confidence)."""
    reply = re.search(r"REPLY:\s*(.*?)\s*CONFIDENCE:", text, re.S)
    conf = re.search(r"CONFIDENCE:\s*(\w+)", text)
    return (reply.group(1).strip() if reply else text.strip(),
            conf.group(1) if conf else "Unknown")


def _field(text: str, name: str) -> str:
    """The value after 'NAME:' on its own line ('' if missing). Allows **bold** markdown."""
    match = re.search(rf"^[*\s]*{name}[*\s]*:[*\s]*(.+?)[*\s]*$", text, re.M | re.I)
    return match.group(1) if match else ""


# ---------- The Log Agent: plain Python, no LLM ----------
class LogAgent(BaseChatAgent):
    """Reads the other agents' answers straight from the conversation and saves the ticket.
    Copying is bookkeeping, not thinking, so no LLM is needed: it can't mistype the reply,
    and it saves two LLM calls per ticket."""

    def __init__(self, complaint: str, customer_name: str = "", order_id: str = ""):
        super().__init__(
            name="log_agent",
            description="Records the ticket to the log file and decides whether to escalate to a human.",
        )
        # the real ticket, as typed by the customer
        self._complaint = complaint
        self._customer_name = customer_name.strip() or "N/A"
        self._order_id = order_id.strip() or "N/A"

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[BaseChatMessage],
                          cancellation_token: CancellationToken) -> Response:
        latest = {m.source: m.to_text() for m in messages}  # last message of each agent
        query = latest.get("query_agent", "")
        reply, confidence = parse_reply(latest.get("resolution_agent", ""))

        # a missing field arrives as "" and the field-validation guardrail escalates it
        row = await asyncio.to_thread(  # file writes + moderation call: keep them off the event loop
            log_ticket,
            complaint=self._complaint,
            category=_field(query, "CATEGORY"),
            urgency=_field(query, "URGENCY"),
            sentiment=_field(query, "SENTIMENT"),
            flags=_field(query, "FLAGS"),
            kb_sources=_field(latest.get("knowledge_agent", ""), "KB_SOURCES"),
            response=reply,
            confidence=confidence,
            language=_field(query, "LANGUAGE") or "English",
            customer_name=self._customer_name,
            order_id=self._order_id,
        )
        content = (f"TICKET: {row['ticket_id']}\n"
                   f"STATUS: {row['status']}\n"
                   f"ESCALATION_REASON: {row['escalation_reason'] or 'none'}\n"
                   "LOGGED")
        return Response(chat_message=TextMessage(content=content, source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass  # keeps no memory between tickets
