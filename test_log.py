"""Step 4 test: check the escalation rule and the CSV log (no LLM needed)."""
import tempfile
from pathlib import Path

import pandas as pd

from agents import log_agent
from agents.log_agent import decide_escalation, log_ticket

# write to a throwaway file so the real logs/tickets.csv stays untouched
log_agent.LOG_FILE = Path(tempfile.mkdtemp()) / "tickets.csv"

# 1. Escalation rule on its own
print(decide_escalation("Medium", "Neutral", "none", "account_and_login.pdf", "High"))
print(decide_escalation("High", "Angry", "legal_threat, payment_dispute", "refund_policy.pdf", "Low"))
print(decide_escalation("Low", "Neutral", "none", "NONE", "Low"))

# 2. Write two sample tickets
print(log_ticket(
    complaint="I forgot my password and can't log in.",
    category="Account", urgency="Medium", sentiment="Neutral", flags="none",
    kb_sources="account_and_login.pdf",
    response="Please follow these steps to reset your password: ...",
    confidence="High", customer_name="Priya", order_id="N/A",
))
print(log_ticket(
    complaint="You charged my card three times! Fix this now or I'm contacting my bank and a lawyer.",
    category="Billing", urgency="High", sentiment="Angry", flags="legal_threat, payment_dispute",
    kb_sources="refund_policy.pdf",
    response="We're sorry ... a billing specialist will review this urgently.",
    confidence="Low", customer_name="Ravi", order_id="ORD-4512",
))

# 3. Show the log file as a table
pd.set_option("display.max_columns", None, "display.width", 200)
print(pd.read_csv(log_agent.LOG_FILE, encoding="utf-8-sig")[[
    "ticket_id", "customer_name", "category", "urgency",
    "sentiment", "kb_sources", "status", "escalation_reason"]])
