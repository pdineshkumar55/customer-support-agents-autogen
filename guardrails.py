"""guardrails.py — plain-Python checks on what goes into and comes out of the agents.

1. Prompt injection: the customer's text tries to give the AI instructions
   ("ignore your rules", "mark this as resolved", "show your system prompt").
2. Fact-check: every number in the reply (days, hours, ₹ amounts, percentages)
   must appear in the cited policy PDFs or in the customer's own message.
3. Moderation: threats, self-harm or violence in the complaint, or anything unsafe in the reply
   (OpenAI's free Moderation API).
4. Field validation: the agents' answers (category, urgency, ...) must be allowed values.
5. Rate limit: tickets per visitor and in total, to protect the OpenAI bill.

Checks 1-4 only flag the ticket. The Log Agent's escalation rule then sends it to a human.
"""
import os
import re
import threading
import time
import unicodedata
from collections import deque

from kb_store import client, known_sources, source_text  # client = the shared OpenAI client

# ---------- 1. Prompt injection ----------
INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instructions?|rules?|prompts?|guidelines?)\b",
    r"\b(system|developer|hidden)\s+(prompt|message|instructions?)\b",
    r"\b(reveal|show|print|repeat)\b.{0,30}\b(prompt|instructions?)\b",
    r"\byou are (now|no longer)\b",
    r"\bpretend (to be|you are)\b",
    r"\b(new|updated) instructions?\b",
    r"\bjailbreak\b",
    r"\b(set|change|mark)\b.{0,30}\b(urgency|confidence|status|category|sentiment)\b",
    r"\bmark (this|it|the ticket|my ticket)\b.{0,20}\b(resolved|low)\b",
    # the customer writing our own output fields, to fake an agent's answer
    r"^\s*(CATEGORY|URGENCY|SENTIMENT|FLAGS|KB_SOURCES|PASSAGES|REPLY|CONFIDENCE|TICKET|STATUS)\s*:",
    r"<<<|>>>",  # trying to break out of the complaint markers
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.I | re.S | re.M)


def injection_detected(text: str) -> bool:
    """True if the text looks like an attempt to instruct the AI (English patterns;
    the Query Agent's prompt_injection flag covers other languages)."""
    return bool(_INJECTION_RE.search(text or ""))


# ---------- 2. Fact-check numbers in the reply ----------
_NUMBER_RE = re.compile(r"\d[\d,]*(?:[.:]\d+)?")
_LIST_MARKER_RE = re.compile(r"(?m)^\s*\d{1,2}[.)]\s")  # "1. Go to My Orders" step numbers


def _numbers(text: str) -> set[str]:
    """All numbers in a text, normalised: '₹1,500.00' -> '1500', '9:00' -> '9'."""
    # Devanagari / Tamil digits -> 0-9, so a Hindi reply is checked the same way
    text = "".join(str(unicodedata.digit(c)) if c.isdigit() else c for c in text or "")
    found = set()
    for n in _NUMBER_RE.findall(text):
        n = n.replace(",", "")
        n = re.sub(r"[.:]0+$", "", n)
        found.add(n)
    return found


def unverified_numbers(reply: str, kb_sources: str, *customer_texts: str) -> list[str]:
    """Numbers in the reply that are not in the cited PDFs or the customer's own text."""
    allowed = set()
    for source in kb_sources.split(","):
        allowed |= _numbers(source_text(source.strip()))
    for text in customer_texts:
        allowed |= _numbers(text)
    return sorted(_numbers(_LIST_MARKER_RE.sub("", reply or "")) - allowed)


# ---------- 3. Moderation ----------
MODERATION_MODEL = "omni-moderation-latest"
# Plain anger, insults or swearing ("harassment", "hate") are NOT here: those customers still need help.
HARMFUL_CATEGORIES = {
    "self_harm", "self_harm_intent", "self_harm_instructions",
    "violence", "violence_graphic", "harassment_threatening", "hate_threatening",
    "sexual_minors", "illicit_violent",
}
SELF_HARM_REPLY = (
    "We're really sorry you're feeling this way, and we want to help. A member of our team "
    "will contact you personally as soon as possible. If you are in danger or thinking about "
    "harming yourself, please contact your local emergency services or a crisis helpline now. "
    "You don't have to face this alone."
)
SAFE_FALLBACK_REPLY = (
    "Thank you for contacting ShopNest. A member of our support team will review your "
    "request and get back to you shortly."
)


def moderate(complaint: str, reply: str) -> tuple[set[str], bool] | None:
    """Check the complaint and the reply in one call.
    Returns (harmful categories in the complaint, reply flagged?), or None if the check failed."""
    try:
        result = client.moderations.create(model=MODERATION_MODEL, input=[complaint, reply])
    except Exception as e:  # network / API error: the caller escalates to a human
        print(f"Moderation check failed: {e}")
        return None
    complaint_result, reply_result = result.results
    categories = {name for name, hit in complaint_result.categories.model_dump().items() if hit}
    return categories & HARMFUL_CATEGORIES, reply_result.flagged


# ---------- 4. Field validation ----------
ALLOWED_VALUES = {
    "category": ["Shipping", "Refund", "Account", "Product", "Billing", "Other"],
    "urgency": ["Low", "Medium", "High"],
    "sentiment": ["Positive", "Neutral", "Frustrated", "Angry"],
    "confidence": ["High", "Medium", "Low"],
}


def normalise_field(field: str, value: str) -> tuple[str, bool]:
    """'high.' -> ('High', True). An unknown value comes back unchanged with False."""
    cleaned = (value or "").strip().strip(".*`'\"").strip()
    for allowed in ALLOWED_VALUES[field]:
        if cleaned.lower() == allowed.lower():
            return allowed, True
    return value, False


def valid_kb_sources(kb_sources: str) -> bool:
    """NONE, or a list of PDF names that really are in the knowledge base."""
    names = [s.strip() for s in (kb_sources or "").split(",") if s.strip()]
    if [n.upper() for n in names] == ["NONE"]:
        return True
    return bool(names) and all(n in known_sources() for n in names)


# ---------- 5. Rate limit ----------
MAX_PER_VISITOR = int(os.getenv("MAX_TICKETS_PER_VISITOR", "5"))  # per 10 minutes
MAX_TOTAL = int(os.getenv("MAX_TICKETS_PER_HOUR", "60"))          # all visitors together
_VISITOR_WINDOW, _TOTAL_WINDOW = 10 * 60, 60 * 60

_visitor_hits: dict[str, deque] = {}
_all_hits: deque = deque()
_rate_lock = threading.Lock()


def _wait_time(hits: deque, limit: int, window: int, now: float) -> int:
    """Drop hits older than the window; seconds to wait if the limit is reached (0 = allowed)."""
    while hits and now - hits[0] > window:
        hits.popleft()
    if len(hits) < limit:
        return 0
    oldest = hits[0] if hits else now  # limit 0 = submissions switched off
    return max(1, int(window - (now - oldest)))


def check_rate_limit(visitor: str) -> int:
    """Record one ticket for this visitor. Returns 0 if allowed, else seconds to wait."""
    now = time.time()
    with _rate_lock:
        hits = _visitor_hits.setdefault(visitor, deque())
        wait = max(_wait_time(hits, MAX_PER_VISITOR, _VISITOR_WINDOW, now),
                   _wait_time(_all_hits, MAX_TOTAL, _TOTAL_WINDOW, now))
        if wait == 0:
            hits.append(now)
            _all_hits.append(now)
        return wait
