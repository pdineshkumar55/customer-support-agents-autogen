"""Step 3 test: run the 3 agents one after another for a single complaint."""
import asyncio
import sys

from llm import get_model_client
from agents.query_agent import create_query_agent
from agents.knowledge_agent import create_knowledge_agent
from agents.resolution_agent import create_resolution_agent


async def main(complaint: str):
    client = get_model_client()
    query_agent = create_query_agent(client)
    knowledge_agent = create_knowledge_agent(client)
    resolution_agent = create_resolution_agent(client)

    print("COMPLAINT:", complaint)

    # 1. Query Agent
    result = await query_agent.run(task=complaint)
    analysis = result.messages[-1].content
    print("\n--- query_agent ---\n" + analysis)

    # 2. Knowledge Agent (gets the complaint + analysis)
    result = await knowledge_agent.run(task=f"Complaint: {complaint}\n\n{analysis}")
    for msg in result.messages:
        if msg.type == "ToolCallRequestEvent":
            print("\n[tool call]", msg.content[0].name, msg.content[0].arguments)
    knowledge = result.messages[-1].content
    print("\n--- knowledge_agent ---\n" + knowledge)

    # 3. Resolution Agent (gets everything so far)
    result = await resolution_agent.run(
        task=f"Complaint: {complaint}\n\n{analysis}\n\n{knowledge}"
    )
    print("\n--- resolution_agent ---\n" + result.messages[-1].content)

    await client.close()


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "I forgot my password and can't log in."
    asyncio.run(main(text))