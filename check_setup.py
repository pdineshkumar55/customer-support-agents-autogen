import os
from dotenv import load_dotenv
import autogen_agentchat, faiss, streamlit

load_dotenv()
print("AutoGen:", autogen_agentchat.__version__)
print("FAISS OK, Streamlit:", streamlit.__version__)
print("API key loaded:", bool(os.getenv("OPENAI_API_KEY")))