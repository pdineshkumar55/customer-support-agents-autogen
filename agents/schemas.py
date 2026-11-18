"""Structured outputs of the Query and Resolution Agents.

The agents use these as `output_content_type`, so the OpenAI API is asked for JSON that
matches the schema (enums included), and AutoGen validates it with pydantic before the
message is sent. A category like "Fraud" or a confidence like "high." can no longer
reach the Log Agent.
"""
from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["Shipping", "Refund", "Account", "Product", "Billing", "Other"]
Urgency = Literal["Low", "Medium", "High"]
Sentiment = Literal["Positive", "Neutral", "Frustrated", "Angry"]
Flag = Literal["legal_threat", "fraud", "payment_dispute", "prompt_injection"]
Confidence = Literal["High", "Medium", "Low"]


class QueryAnalysis(BaseModel):
    """What the Query Agent works out from the complaint."""
    category: Category
    urgency: Urgency
    sentiment: Sentiment
    flags: list[Flag] = Field(description="Every flag that applies; empty list if none.")
    summary: str = Field(description="One short sentence describing the problem, in English.")
    language: str = Field(description='Language the customer wrote in, e.g. "English", '
                                      '"Tamil", "Tamil (Latin script)".')

    def flags_text(self) -> str:
        """'legal_threat, payment_dispute', or 'none' (the format of the log's flags column)."""
        return ", ".join(self.flags) or "none"


class Resolution(BaseModel):
    """The Resolution Agent's reply to the customer."""
    reply: str = Field(description="The message to the customer.")
    confidence: Confidence


# readable text for the UI timeline and for the next agents in the conversation
QUERY_FORMAT = ("CATEGORY: {category}\nURGENCY: {urgency}\nSENTIMENT: {sentiment}\n"
                "FLAGS: {flags}\nSUMMARY: {summary}\nLANGUAGE: {language}")
RESOLUTION_FORMAT = "REPLY:\n{reply}\nCONFIDENCE: {confidence}"
