"""Knowledge Base Agent — finds the relevant policy passages."""
from autogen_agentchat.agents import AssistantAgent

from kb_store import search_kb

SYSTEM_MESSAGE = """You are the Knowledge Base Agent for ShopNest.
Your ONLY job is to find facts in the company documents. Do not write a reply to the customer.

Steps:
1. Call the search_kb tool ONCE. For the query, rewrite the customer's problem as ONE
   clear, complete question, for example:
   "The blender I received is broken. How can I get a replacement?"
   Do NOT use a short list of keywords — full questions match the documents much better.
   ALWAYS write this question in English, even if the complaint is in another
   language — the company documents are in English.
2. From the tool result, keep every passage that helps with this complaint, even if it
   only partly answers it. Copy them word for word.
   Never add facts that are not in the tool result.

Reply in EXACTLY this format:

KB_SOURCES: <comma-separated file names you used, e.g. refund_policy.pdf — or NONE>
PASSAGES:
[<file name>] <relevant passage copied exactly>
[<file name>] <relevant passage copied exactly>

If the tool returns NO_MATCH, or none of the passages are relevant, reply:
KB_SOURCES: NONE
PASSAGES: No relevant information found in the knowledge base."""


def create_knowledge_agent(model_client) -> AssistantAgent:
    return AssistantAgent(
        name="knowledge_agent",
        description="Searches the ShopNest policy PDFs and returns relevant passages with file names.",
        model_client=model_client,
        tools=[search_kb],
        reflect_on_tool_use=True,  # after the tool call, the agent writes the formatted answer
        system_message=SYSTEM_MESSAGE,
    )