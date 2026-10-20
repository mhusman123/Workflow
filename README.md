# Cortex AI — Conversational Agent with LangGraph & Groq

<p align="center">
  <img src="https://img.shields.io/badge/LangGraph-0.2+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.33+-red?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Groq-API-orange?logo=groq&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.10+-yellow?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

> **Cortex** is a multi-turn conversational AI agent built with [LangGraph](https://langchain-ai.github.io/langgraph/) and served through a ChatGPT-style [Streamlit](https://streamlit.io) frontend. It uses the [Groq API](https://groq.com/) for fast hosted LLM inference.


---

## ✨ Features

- 🧠 **Stateful multi-turn memory** — conversations persist across messages using LangGraph's `InMemorySaver` checkpointer
- 💬 **Multi-session support** — create, rename, delete, and switch between multiple chat threads
- ⚡ **Streaming responses** — tokens stream in real-time via `chatbot.stream()`
- 🎨 **ChatGPT-style UI** — dark sidebar, hero landing, suggestion cards, and clean chat feed
- 🔒 **Fully local** — runs 100% on your machine, zero data sent to the cloud
- 🖥️ **Responsive** — desktop hover-controls + mobile long-press (650ms) for chat actions

---


## Project Vision

This repository is a practical, progressively expanding guide to building Agentic AI systems with LangGraph. The current implementation is the foundation: a stateful conversational agent with a Streamlit interface, local model execution, graph-based orchestration, streaming responses, and per-thread memory.

The project will grow from this core into a collection of focused implementations covering the major patterns used in production agent systems. Each addition will remain runnable, explainable, and connected to the LangGraph concepts it demonstrates.

## What We Use Today

- **LangGraph** for explicit state, nodes, edges, graph execution, and checkpointing
- **LangChain Core** for message types and model interaction contracts
- **Groq** for hosted LLM inference through `ChatGroq`
- **Streamlit** for the interactive chat application and streamed responses
- **Python** and `python-dotenv` for the application runtime and configuration
- **InMemorySaver** for thread-scoped conversation checkpoints during development

## Agentic AI Topics Covered and Planned

The repository is intended to cover the full LangGraph Agentic AI workflow, from fundamentals to production concerns.

| Area | Status | Planned implementation |
|---|---|---|
| Graph fundamentals | In progress | State schemas, nodes, edges, entry points, and graph execution |
| Stateful conversations | Current | Multi-turn messages, thread IDs, streaming, and checkpoints |
| Tool calling | Planned | Structured tools, tool nodes, routing, and tool-result handling |
| ReAct agents | Planned | Reasoning and action loops with bounded execution |
| Human-in-the-loop | Planned | Interrupts, approvals, review steps, and resumable runs |
| Memory and persistence | Planned | Long-term memory, SQLite/Postgres checkpointers, and recovery |
| RAG agents | Planned | Ingestion, retrieval, grounded responses, and corrective retrieval |
| Multi-agent systems | Planned | Supervisor, handoff, collaboration, and specialist graphs |
| MCP integration | Planned | External tools and data sources through Model Context Protocol |
| Structured output | Planned | Schemas, validation, retries, and reliable agent responses |
| Observability and evaluation | Planned | LangSmith traces, datasets, evaluations, latency, and cost tracking |

## 🏗️ Architecture

```
User (Browser)
      │
      ▼
┌─────────────────────────────────────────────┐
│          Streamlit Frontend                  │
│   streamlit_frontend_thread.py               │
│                                              │
│  • Session state management (threads)        │
│  • Streaming UI via st.write_stream()        │
│  • Sidebar: New Chat / Rename / Delete       │
│  • Hero landing + suggestion cards           │
└─────────────────┬───────────────────────────┘
                  │  chatbot.stream()
                  ▼
┌─────────────────────────────────────────────┐
│          LangGraph Backend                   │
│   langgraph_backend.py                       │
│                                              │
│  StateGraph(ChatState)                       │
│    START → chat_node → END                   │
│                                              │
│  InMemorySaver (per thread_id checkpoint)    │
└─────────────────┬───────────────────────────┘
                  │  llm.invoke(messages)
                  ▼
┌─────────────────────────────────────────────┐
│             Groq API                         │
│   Model: openai/gpt-oss-20b                  │
│   API key: GROQ_API_KEY                      │
└─────────────────────────────────────────────┘
```

### Data Flow

1. User types a message → `streamlit_frontend_thread.py` appends it to `session_state["message_history"]`
2. A `HumanMessage` is sent to `chatbot.stream()` with the current `thread_id` as config
3. LangGraph routes through `chat_node` → Groq LLM → returns `AIMessage`
4. Tokens stream back to the UI via `st.write_stream()`
5. The full response is stored in session state and LangGraph's checkpointer for memory persistence

---

## 📁 Folder Structure

```
Agent-using-LangGraph/
├── langgraph_backend.py          # LangGraph graph definition, LLM, checkpointer
├── streamlit_frontend_thread.py  # Main Streamlit UI (multi-thread chatbot)
├── streamlit_frontend.py         # (Legacy) single-session frontend
├── basic_chatbot.ipynb           # Jupyter notebook: basic chatbot exploration
├── simple_llm_workflow.ipynb     # Jupyter notebook: simple LLM workflow demo
├── requirements.txt              # Python dependencies
├── .env                          # Local secrets (GROQ_API_KEY, GROQ_MODEL)
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## 🧱 Tech Stack

| Layer | Technology | Version | Link |
|---|---|---|---|
| **Agent Orchestration** | LangGraph | `>=0.2` | [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/) |
| **LLM Framework** | LangChain Core | `>=0.2` | [python.langchain.com](https://python.langchain.com/) |
| **LLM Provider** | Groq API | hosted | [groq.com](https://groq.com/) |
| **LLM Model** | openai/gpt-oss-20b | configurable | [Groq Models](https://console.groq.com/docs/models) |
| **Groq Integration** | langchain-groq | `>=0.2` | [PyPI: langchain-groq](https://pypi.org/project/langchain-groq/) |
| **Frontend** | Streamlit | `>=1.33` | [streamlit.io](https://streamlit.io/) |
| **Env Config** | python-dotenv | `>=1.0` | [pypi: python-dotenv](https://pypi.org/project/python-dotenv/) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A Groq account and API key from the [Groq Console](https://console.groq.com/keys)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/mhusman123/Agent-using-LangGraph.git
cd Agent-using-LangGraph

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure Groq locally. Keep this file private and never commit it.
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=openai/gpt-oss-20b
```

### Run

```bash
streamlit run streamlit_frontend_thread.py
```

Open your browser at **http://localhost:8501**

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | required | Groq API key; use Streamlit Secrets when deployed |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Any supported Groq chat model |

Swap the model anytime in `.env` or Streamlit Secrets:

```env
GROQ_MODEL=openai/gpt-oss-20b
```

For Streamlit Community Cloud, add the following under **Settings > Secrets**:

```toml
GROQ_API_KEY = "your-groq-api-key"
GROQ_MODEL = "openai/gpt-oss-20b"
```

---

## 🔬 LangGraph Graph Structure

```python
# langgraph_backend.py

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=InMemorySaver())
```

Each conversation thread gets its own **checkpoint** identified by `thread_id`, giving full per-session memory without any database.

---

## 🗺️ Roadmap — What We Are Building Next

This project is the **foundation layer** of a fully-featured AI agent platform. Here is what is coming:

### 🔭 Observability — LangSmith Tracing
- Integrate [LangSmith](https://smith.langchain.com/) for full trace visibility: every LLM call, node execution, latency, and token count — all in one dashboard
- Enable `LANGCHAIN_TRACING_V2=true` for automatic graph tracing
- Docs: [docs.smith.langchain.com](https://docs.smith.langchain.com/)

### 🤝 Human-in-the-Loop (HITL)
- Use LangGraph's [`interrupt()`](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/) to pause graph execution and request human approval before sensitive steps
- Build approval flows: tool calls, code execution, data writes
- Docs: [Human-in-the-loop guide](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)

### 🔌 MCP Client (Model Context Protocol)
- Connect to external tools and data sources via [MCP](https://modelcontextprotocol.io/) clients
- Plug in filesystem access, browser control, API clients, and more — all as LangGraph tool nodes
- Docs: [modelcontextprotocol.io](https://modelcontextprotocol.io/introduction)

### 📚 RAG — Retrieval-Augmented Generation
- Add a vector store (e.g., [Chroma](https://www.trychroma.com/), [FAISS](https://faiss.ai/)) to give the agent long-term knowledge
- Use [LangChain document loaders](https://python.langchain.com/docs/integrations/document_loaders/) to ingest PDFs, web pages, and code files
- Docs: [RAG with LangGraph](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/)

### 🗄️ SQLite Persistence
- Replace `InMemorySaver` with [LangGraph's SQLite checkpointer](https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.sqlite.SqliteSaver) for durable, cross-session memory
- Store chat history, thread metadata, and user preferences in a local SQLite database
- Docs: [Persistence in LangGraph](https://langchain-ai.github.io/langgraph/concepts/persistence/)

### 🕵️ Research Agent
- Build a multi-step autonomous research agent using [LangGraph ReAct](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/) pattern
- Tools: web search, document reader, summarizer, citation generator
- Streaming thought-chain visibility in the UI

### 🛠️ Tool Use and Function Calling
- Bind custom Python functions as tools using `llm.bind_tools()`
- Add tools: web search, calculator, code interpreter, file reader
- Docs: [Tool calling in LangChain](https://python.langchain.com/docs/concepts/tool_calling/)

### 👤 Auth and Multi-User Support
- User authentication with [Streamlit-Authenticator](https://github.com/mkhorasani/Streamlit-Authenticator)
- Per-user thread isolation and history

---

## 📖 Key References

- 📘 [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- 📘 [LangGraph Quickstart](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- 📘 [LangGraph How-To Guides](https://langchain-ai.github.io/langgraph/how-tos/)
- 📘 [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/)
- 📘 [Groq API Documentation](https://console.groq.com/docs)
- 📘 [Streamlit API Reference](https://docs.streamlit.io/library/api-reference)
- 📘 [LangSmith Observability](https://docs.smith.langchain.com/)
- 📘 [Model Context Protocol](https://modelcontextprotocol.io/introduction)

---

## 📄 License

MIT © [mhusman123](https://github.com/mhusman123)

---

<p align="center">Built with ❤️ using <a href="https://langchain-ai.github.io/langgraph/">LangGraph</a> · <a href="https://streamlit.io">Streamlit</a> · <a href="https://groq.com/">Groq</a></p>
