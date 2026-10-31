# 🛎️ AI Customer Support Desk — Multi-Agent System with AutoGen

A team of four AI agents, built with **AutoGen**, that handles customer complaints for **ShopNest**, a made-up online store. The agents work out what the problem is, find the answer in the company's own policy documents, write a grounded reply, and log every ticket. Hard cases are escalated to a human.

Built for the **Social Eagle AI Engineering Program — Multi-Agent Systems Track** (Buildathon assignment).

![Escalated ticket](screenshots/escalated_ticket.png)

---

## ✨ Features

- **4 AutoGen agents**, each with one clear job, running in a fixed order with `RoundRobinGroupChat`
- **Grounded answers (RAG):** policy PDFs → chunks → OpenAI embeddings → **FAISS** vector search *(bonus)*
- **Shows its sources:** every reply lists the PDF file(s) it used
- **No made-up policies:** if nothing relevant is found, the reply says so honestly and the ticket is escalated
- **Rule-based escalation** applied in Python, so it is the same every time
- **CSV ticket log** with every column required by the assignment
- **Streamlit UI:** live agent timeline, green/red status badge, highlighted reply, log viewer with download

---

## 🧠 Architecture

```mermaid
flowchart LR
    U[Customer complaint<br/>Streamlit form] --> Q[🔍 Query Agent<br/>category · urgency · sentiment · flags]
    Q --> K[📚 Knowledge Agent<br/>search_kb tool]
    K <--> F[(FAISS index<br/>4 policy PDFs)]
    K --> R[✍️ Resolution Agent<br/>grounded reply + confidence]
    R --> L[🗂️ Log Agent<br/>log_ticket tool]
    L --> C[(logs/tickets.csv)]
    L --> D{Escalation rule}
    D -->|any rule true| E[🚨 Escalated]
    D -->|none true| S[✅ Resolved]
```

| Agent | Job | Tool | Output |
|---|---|---|---|
| **Customer Query Agent** | Understands the complaint | — | `CATEGORY`, `URGENCY`, `SENTIMENT`, `FLAGS`, `SUMMARY` |
| **Knowledge Base Agent** | Finds the relevant policy passages | `search_kb(query)` | `KB_SOURCES` + passages copied word for word |
| **Resolution Agent** | Writes a polite reply using only those passages | — | `REPLY` + `CONFIDENCE` (High/Medium/Low) |
| **Log Agent** | Records the ticket and gets the escalation decision | `log_ticket(...)` | Ticket ID, status, reason, then `LOGGED` |

**Team and termination:** `RoundRobinGroupChat` runs Query → Knowledge → Resolution → Log. It stops when the Log Agent says **`LOGGED`** (`TextMentionTermination`, limited to `log_agent`), with `MaxMessageTermination(10)` as a safety net.

### 🚨 Escalation rule

A ticket is **Escalated** if **any** of these is true. Otherwise it is **Resolved**.

| Rule | Comes from |
|---|---|
| Urgency is **High** | Query Agent |
| Sentiment is **Angry** | Query Agent |
| **No matching document** (`KB_SOURCES: NONE`) | Knowledge Agent |
| **Legal threat**, **fraud** or **payment dispute** flag | Query Agent |
| Resolution **confidence is Low** | Resolution Agent |

The rule is plain Python (`decide_escalation()` in `agents/log_agent.py`), not left to the model, so it is predictable and easy to test. The reason lists every rule that matched, for example `High urgency + Angry customer + Legal threat + Payment dispute + Low confidence`.

---

## 🛠️ Tech stack

| Layer | Tool |
|---|---|
| Agents | AutoGen `autogen-agentchat` 0.7 (`AssistantAgent`, `RoundRobinGroupChat`) |
| LLM | OpenAI `gpt-4o-mini` via `autogen-ext[openai]` |
| Knowledge base | 4 policy **PDFs** → `pypdf` → chunks → `text-embedding-3-small` → **FAISS** |
| Logging | Python `csv` (+ `pandas` for viewing) |
| UI | Streamlit |

---

## 📁 Project structure

```
customer-support-agents-autogen/
├── app.py                    # Streamlit UI
├── team.py                   # wires the 4 agents into one AutoGen team
├── llm.py                    # shared OpenAI model client
├── kb_store.py               # PDF reading, chunking, FAISS index, search_kb tool
├── agents/
│   ├── query_agent.py        # Customer Query Agent
│   ├── knowledge_agent.py    # Knowledge Base Agent (uses search_kb)
│   ├── resolution_agent.py   # Resolution Agent
│   └── log_agent.py          # Log Agent + log_ticket tool + escalation rule
├── knowledge_base/
│   ├── shipping_policy.pdf
│   ├── refund_policy.pdf
│   ├── account_and_login.pdf
│   └── product_faq.pdf
├── logs/tickets.csv          # created automatically
├── faiss_index/              # created automatically (git-ignored)
├── screenshots/
├── test_kb.py · test_agents.py · test_log.py · test_team.py   # step-by-step tests
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Setup and run

**Requirements:** Python 3.10+ and an OpenAI API key.

```bash
git clone https://github.com/pdineshkumar55/customer-support-agents-autogen.git
cd customer-support-agents-autogen

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your key:

```
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
KB_MIN_SCORE=0.30
```

Run the app:

```bash
streamlit run app.py
```

The FAISS index is built automatically on the first search and saved to `faiss_index/`. It is rebuilt only when a PDF is added or changed.

**Optional command-line tests:**

```bash
python test_kb.py        # retrieval scores for the 6 test complaints
python test_log.py       # escalation rule + CSV (no LLM calls)
python test_team.py      # all 6 complaints through the full team
python test_team.py 5    # just test case 5
```

---

## 🧪 Test results

| # | Complaint | Category · Urgency · Sentiment | Source | Result |
|---|---|---|---|---|
| 1 | Where is my order? It was supposed to arrive yesterday. | Shipping · Medium · Frustrated | shipping_policy.pdf | ✅ Resolved |
| 2 | I want to return the shoes I bought last week… | Refund · Low · Neutral | refund_policy.pdf | ✅ Resolved |
| 3 | I forgot my password and can't log in. | Account · Medium · Neutral | account_and_login.pdf | ✅ Resolved |
| 4 | The blender I received is broken… | Product · Medium · Frustrated | product_faq.pdf, refund_policy.pdf | ✅ Resolved |
| 5 | You charged my card three times!… bank and a lawyer. | Billing · High · Angry | refund_policy.pdf | 🚨 Escalated: High urgency + Angry customer + Legal threat + Payment dispute + Low confidence |
| 6 | Do you sell gift cards for a company of 500 employees…? | Other · Low · Neutral | NONE | 🚨 Escalated: No matching document + Low confidence |
| 7 | *(your own test case)* | | | |
| 8 | *(your own test case)* | | | |

---

## 📸 Screenshots

| Resolved ticket | Escalated ticket |
|---|---|
| ![Resolved](screenshots/resolved_ticket.png) | ![Escalated](screenshots/escalated_ticket.png) |

**Ticket log**

![Ticket log](screenshots/ticket_log.png)

**My own test cases**

| Test case 7 | Test case 8 |
|---|---|
| ![Custom 1](screenshots/custom_test_1.png) | ![Custom 2](screenshots/custom_test_2.png) |

---

## 📋 Log file format (`logs/tickets.csv`)

One row per ticket:

`ticket_id, timestamp, customer_name, order_id, complaint, category, urgency, sentiment, flags, kb_sources, response, confidence, status, escalation_reason`

Example: `TKT-0003 · Ravi · ORD-1187 · Billing · High · Angry · refund_policy.pdf · Escalated · High urgency + Angry customer + Legal threat + Payment dispute + Low confidence`

---

## 💡 Lessons learned (hardest problems solved)

1. **Short keyword queries retrieve much worse than full sentences.** The Knowledge Agent first searched `"broken blender replacement"`, which scored **0.23** and fell below the cut-off, so the ticket was wrongly escalated. Rewriting the search as a full question (*"The blender I received is broken. How can I get a replacement?"*) scored **0.43** and found the right passages. The agent's instructions now require a full question.
2. **FAISS always returns something.** That makes a similarity cut-off (`KB_MIN_SCORE`) essential for detecting "no matching document". I tuned it from the real scores: real matches scored 0.32–0.57, the out-of-scope gift-card question scored about 0.20, so **0.30** separates them cleanly. At 0.40, useful passages (such as the password reset steps, at 0.39) were dropped.
3. **Tables in PDFs break retrieval.** Extracted tables came out as loose cells ("15 days" no longer next to "Footwear"). Writing each table row as one full sentence in the PDFs fixed it.
4. **Escalation belongs in code, not in a prompt.** The model extracts the facts; Python applies the rule. The decision is consistent and testable without calling the LLM (`test_log.py`).
5. **Safe failure:** even when retrieval failed, the system escalated with an honest reply instead of making up a policy.

---

## 🔭 Possible next steps

- `SelectorGroupChat` so the model decides who speaks next
- A human-in-the-loop approval for escalated replies (`UserProxyAgent`)
- A dashboard: tickets per category, resolved vs escalated over time
- Detect the customer's language and reply in the same language
- Deploy with Docker on a VPS

---

## 🔐 Security

API keys live only in `.env`, which is listed in `.gitignore` and never committed. `.env.example` shows the variable names without values.