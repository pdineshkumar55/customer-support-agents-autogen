"""The eight test complaints shared by test_kb.py, test_log.py and test_team.py.

Cases 1-6 are the assignment brief's complaints; cases 7-8 are our own.
`agent_output` is what the Query, Knowledge and Resolution Agents answered in the
recorded runs (README "Test results"), so test_log.py can check the escalation rule
without calling an LLM.
"""
from dataclasses import dataclass, field

import pytest


@dataclass(frozen=True)
class Case:
    id: str
    customer_name: str
    order_id: str
    complaint: str
    category: str      # expected category
    status: str        # expected status: Resolved / Escalated
    top_source: str    # expected best FAISS match, or "NONE"
    agent_output: dict = field(default_factory=dict)


def _output(urgency, sentiment, flags, kb_sources, confidence):
    return dict(urgency=urgency, sentiment=sentiment, flags=flags,
                kb_sources=kb_sources, confidence=confidence)


CASES = [
    Case("1-late-order", "Priya", "ORD-4512",
         "Where is my order? It was supposed to arrive yesterday.",
         "Shipping", "Resolved", "shipping_policy.pdf",
         _output("Medium", "Frustrated", "none", "shipping_policy.pdf", "High")),
    Case("2-refund", "Arjun", "ORD-3301",
         "I want to return the shoes I bought last week. How do I get a refund?",
         "Refund", "Resolved", "refund_policy.pdf",
         _output("Low", "Neutral", "none", "refund_policy.pdf", "High")),
    Case("3-password", "Meena", "",
         "I forgot my password and can't log in.",
         "Account", "Resolved", "account_and_login.pdf",
         _output("Medium", "Neutral", "none", "account_and_login.pdf", "High")),
    Case("4-broken-blender", "Karthik", "ORD-2290",
         "The blender I received is broken. Can I get a replacement?",
         "Product", "Resolved", "product_faq.pdf",
         _output("Medium", "Frustrated", "none", "product_faq.pdf, refund_policy.pdf", "High")),
    Case("5-charged-three-times", "Ravi", "ORD-1187",
         "You charged my card three times! Fix this now or I'm contacting my bank and a lawyer.",
         "Billing", "Escalated", "refund_policy.pdf",
         _output("High", "Angry", "legal_threat, payment_dispute", "refund_policy.pdf", "Low")),
    Case("6-corporate-gift-cards", "Divya", "",
         "Do you sell gift cards for a company of 500 employees with custom branding?",
         "Other", "Escalated", "NONE",
         _output("Low", "Neutral", "none", "NONE", "Low")),
    # ----- our own cases -----
    Case("7-change-address", "Anand", "ORD-5120",
         "I want to change my delivery address, the order hasn't shipped yet.",
         "Shipping", "Resolved", "shipping_policy.pdf",
         _output("Low", "Neutral", "none", "shipping_policy.pdf", "High")),
    Case("8-unknown-orders", "Kavya", "",
         "Someone placed 2 orders from my account that I didn't make.",
         "Account", "Escalated", "account_and_login.pdf",
         _output("High", "Angry", "fraud", "account_and_login.pdf", "Low")),
]

# use as: @pytest.mark.parametrize("case", CASE_PARAMS)
CASE_PARAMS = [pytest.param(c, id=c.id) for c in CASES]
