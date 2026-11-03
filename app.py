"""app.py — Streamlit UI for the ShopNest AI Customer Support Desk.

Run with:  streamlit run app.py
"""
import asyncio
import html
import re

import pandas as pd
import streamlit as st
from autogen_agentchat.base import TaskResult

from agents.log_agent import LOG_FILE
from dashboard import render_dashboard
from llm import get_model_client
from team import build_task, create_team

st.set_page_config(page_title="ShopNest AI Support Desk", page_icon="🛎️", layout="wide")

AGENTS = {
    "query_agent": ("🔍", "Customer Query Agent", "Category, urgency, sentiment"),
    "knowledge_agent": ("📚", "Knowledge Base Agent", "Searches the policy PDFs"),
    "resolution_agent": ("✍️", "Resolution Agent", "Writes the customer reply"),
    "log_agent": ("🗂️", "Log Agent", "Logs the ticket, decides escalation"),
}

SAMPLES = {
    "— type your own —": ("", "", ""),
    "Late order": ("Priya", "ORD-4512", "Where is my order? It was supposed to arrive yesterday."),
    "Return shoes": ("Arjun", "ORD-3301", "I want to return the shoes I bought last week. How do I get a refund?"),
    "Forgot password": ("Meena", "", "I forgot my password and can't log in."),
    "Broken blender": ("Karthik", "ORD-2290", "The blender I received is broken. Can I get a replacement?"),
    "Charged 3 times": ("Ravi", "ORD-1187", "You charged my card three times! Fix this now or I'm contacting my bank and a lawyer."),
    "Corporate gift cards": ("Divya", "", "Do you sell gift cards for a company of 500 employees with custom branding?"),
    "Tamil: broken blender": ("Karthik", "ORD-2290", "நான் வாங்கிய பிளெண்டர் உடைந்த நிலையில் வந்தது. எனக்கு மாற்றுப் பொருள் கிடைக்குமா?"),
    "Hindi: forgot password": ("Anita", "", "मैं अपना पासवर्ड भूल गई हूँ और लॉगिन नहीं कर पा रही हूँ।"),
    "Tanglish: late order": ("Selvi", "ORD-5120", "En order innum varala, nethu vandhurum nu sonninga. Enna aachu?"),
}

st.markdown("""
<style>
.badge {display:inline-block;padding:4px 14px;border-radius:999px;font-weight:700;color:#fff;font-size:0.95rem}
.badge-resolved {background:#1a7f37}
.badge-escalated {background:#cf222e}
.chip {display:inline-block;margin:8px 6px 4px 0;padding:4px 12px;border-radius:999px;
       border:1px solid rgba(128,128,128,0.4);font-size:0.9rem}
</style>
""", unsafe_allow_html=True)


# ---------- helpers ----------
def parse_reply(text: str) -> tuple[str, str]:
    """Split the Resolution Agent's message into (reply, confidence)."""
    reply = re.search(r"REPLY:\s*(.*?)\s*CONFIDENCE:", text, re.S)
    conf = re.search(r"CONFIDENCE:\s*(\w+)", text)
    return (reply.group(1).strip() if reply else text.strip(),
            conf.group(1) if conf else "Unknown")


def parse_ticket_id(text: str) -> str:
    """Pull the ticket ID out of the Log Agent's message ('' if missing)."""
    match = re.search(r"TICKET:\s*(TKT-\d+)", text)
    return match.group(1) if match else ""


def load_log() -> pd.DataFrame:
    if LOG_FILE.exists():
        return pd.read_csv(LOG_FILE, encoding="utf-8-sig").fillna("")
    return pd.DataFrame()


async def run_ticket(complaint: str, name: str, order_id: str, timeline) -> dict:
    """Run the team and show each agent's output in the timeline as it happens."""
    client = get_model_client()
    team = create_team(client)
    outputs = {}
    tool_notes = {}

    try:
        async for msg in team.run_stream(task=build_task(complaint, name, order_id)):
            if isinstance(msg, TaskResult):
                break
            source = getattr(msg, "source", "")
            if source not in AGENTS:
                continue
            icon, title, _ = AGENTS[source]
            if msg.type == "ToolCallRequestEvent":
                call = msg.content[0]
                args = call.arguments if len(call.arguments) < 120 else call.arguments[:120] + "…"
                tool_notes.setdefault(source, []).append(f"🔧 `{call.name}` called with `{args}`")
            elif msg.type == "TextMessage":
                outputs[source] = msg.content
                with timeline:
                    with st.expander(f"{icon} {title}", expanded=True):
                        for note in tool_notes.get(source, []):
                            st.caption(note)
                        st.code(msg.content, language=None, wrap_lines=True)
    finally:
        await client.close()
    return outputs


# ---------- sidebar ----------
with st.sidebar:
    st.header("🛎️ ShopNest Support Desk")
    st.write("A team of 4 AutoGen agents handles every complaint:")
    for icon, title, job in AGENTS.values():
        st.markdown(f"{icon} **{title}** — {job}")
    st.divider()
    st.caption("Knowledge base: shipping, refund, account & login, product FAQ (PDF) · FAISS search")

# ---------- main ----------
st.title("AI Customer Support Desk")
tab_new, tab_log, tab_dash = st.tabs(["📝 New Ticket", "📋 Ticket Log", "📊 Dashboard"])

with tab_new:
    sample = st.selectbox("Load a sample complaint (optional)", list(SAMPLES))
    s_name, s_order, s_text = SAMPLES[sample]

    with st.form("ticket_form"):
        c1, c2 = st.columns(2)
        name = c1.text_input("Customer name (optional)", value=s_name, max_chars=100)
        order_id = c2.text_input("Order ID (optional)", value=s_order, max_chars=100)
        complaint = st.text_area("Complaint", value=s_text, height=120, max_chars=2000,
                                 placeholder="Describe the problem…")
        submitted = st.form_submit_button("Submit ticket", type="primary")

    if submitted:
        if not complaint.strip():
            st.warning("Please enter a complaint.")
            st.stop()

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Agent steps")
            timeline = st.container()
        with right:
            st.subheader("Result")
            with st.spinner("The agents are working on this ticket…"):
                try:
                    outputs = asyncio.run(run_ticket(complaint, name, order_id, timeline))
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
                    st.stop()

            reply, confidence = parse_reply(outputs.get("resolution_agent", ""))
            # look the ticket up by its ID, not "last row", so parallel users never mix tickets
            ticket_id = parse_ticket_id(outputs.get("log_agent", ""))
            log = load_log()
            match = log[log["ticket_id"] == ticket_id] if not log.empty else log
            last = match.iloc[-1] if ticket_id and not match.empty else None

            if last is not None:
                status = last["status"]
                css = "badge-resolved" if status == "Resolved" else "badge-escalated"
                label = "✅ Resolved" if status == "Resolved" else "🚨 Escalated to human support"
                st.markdown(f'<span class="badge {css}">{label}</span> &nbsp; '
                            f'**{html.escape(last["ticket_id"])}**', unsafe_allow_html=True)
                if status == "Escalated":
                    st.markdown(f"**Reason:** {last['escalation_reason']}")
                chips = [("Category", last["category"]), ("Urgency", last["urgency"]),
                         ("Sentiment", last["sentiment"]), ("Confidence", confidence),
                         ("Language", last.get("language", "English"))]
                # values come from the LLM, so escape them before rendering as HTML
                st.markdown(" ".join(f'<span class="chip"><b>{k}:</b> {html.escape(str(v))}</span>'
                                     for k, v in chips),
                            unsafe_allow_html=True)
                st.caption(f"📄 Sources: {last['kb_sources']}")
            else:
                st.warning("The ticket was not logged — check the agent steps.")

            st.markdown("**Reply to customer**")
            st.info(reply, icon="💬")

with tab_log:
    log = load_log()
    if log.empty:
        st.info("No tickets yet. Submit one from the New Ticket tab.")
    else:
        total = len(log)
        escalated = int((log["status"] == "Escalated").sum())
        k1, k2, k3 = st.columns(3)
        k1.metric("Total tickets", total)
        k2.metric("Resolved", total - escalated)
        k3.metric("Escalated", escalated)

        status_filter = st.multiselect("Filter by status", ["Resolved", "Escalated"],
                                       default=["Resolved", "Escalated"])
        view = log[log["status"].isin(status_filter)].iloc[::-1]  # newest first
        st.dataframe(view, width="stretch", hide_index=True)

        st.download_button("⬇️ Download log (CSV)",
                           data=log.to_csv(index=False).encode("utf-8-sig"),
                           file_name="tickets.csv", mime="text/csv")

with tab_dash:
    render_dashboard(load_log())