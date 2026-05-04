# Agentis using LangGraph (Ollama + Streamlit)

Small demo project showing how to:
- build a minimal LangGraph chatbot backend using `langchain-ollama`, and
- run an interactive Streamlit chat UI that calls the compiled graph.

## Files

- `langgraph_backend.py`
  - **Backend graph**: defines `ChatState`, a `chat_node`, and compiles the LangGraph into `chatbot`.
  - Uses `InMemorySaver` with a `thread_id` (passed from the frontend) so your conversation can be checkpointed per thread.
  - Uses Ollama via `ChatOllama(model="llama3")`.

- `streamlit_frontend.py`
  - **Frontend UI**: Streamlit chat app.
  - Stores chat messages in `st.session_state["message_history"]`.
  - On each user message, calls `chatbot.invoke({"messages": [HumanMessage(...)]}, config=CONFIG)`.

- `basic_chatbot.ipynb`
  - Notebook version of the basic LangGraph chatbot call.

- `simple_llm_workflow.ipynb`
  - Another notebook experimenting with a simple LLM workflow.

- `test.ipynb`
  - Misc notebook for quick tests.

## Requirements

Python 3.10+.

This project uses **Ollama**. You must have Ollama installed and running locally.

## Setup (Windows / PowerShell)

Create and activate a venv (recommended):

```powershell
cd F:\Wrokflows
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
\.\.venv\Scripts\Activate.ps1
```

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

Ensure Ollama is running and the model is available:

```powershell
ollama pull llama3
```

## Run the app

```powershell
streamlit run .\streamlit_frontend.py
```

Open in your browser:

- http://localhost:8501

## Notes

- Do **not** commit `.venv/` or `myenv/` to GitHub (they are ignored by `.gitignore`).
- If Streamlit shows Ollama connection errors, start Ollama and retry.
