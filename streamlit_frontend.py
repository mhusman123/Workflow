from __future__ import annotations

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from langgraph_backend import (
    MODE_GENERAL,
    MODE_MCP,
    MODE_RAG,
    MODES,
    ensure_thread,
    get_chatbot,
    get_thread_title,
    ingest_pdf,
    rename_thread,
    retrieve_all_threads,
    stream_chat,
    thread_document_metadata,
)


st.set_page_config(
    page_title="Cortex",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --cortex-ink: #2f2d2a; --cortex-muted: #78736d; --cortex-line: #e8e3dc; --cortex-accent: #d97757; }
    .stApp { background: #fbfaf8; color: var(--cortex-ink); }
    [data-testid="stSidebar"] { background: #f4f1ec; border-right: 1px solid var(--cortex-line); }
    [data-testid="stSidebar"] .block-container { padding: 1.4rem 1rem; }
    .main .block-container { max-width: 900px; padding-top: 2rem; padding-bottom: 7rem; }
    .brand { font-size: 1.55rem; font-weight: 700; letter-spacing: -.04em; margin-bottom: 1.2rem; }
    .brand span { color: var(--cortex-accent); }
    .welcome { text-align: center; padding: 18vh 1rem 2rem; }
    .welcome h1 { font-size: 2.7rem; letter-spacing: -.05em; font-weight: 650; }
    .welcome p { color: var(--cortex-muted); font-size: 1.05rem; }
    .stChatMessage { background: transparent !important; padding: 1.2rem 0 !important; border-bottom: 1px solid rgba(232,227,220,.7); }
    .stChatMessage [data-testid="stChatMessageContent"] { line-height: 1.7; }
    div[data-testid="stSidebar"] button { border-radius: 8px; }
    .disclaimer { color: var(--cortex-muted); text-align: center; font-size: .78rem; margin-top: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def new_thread_id() -> str:
    return str(uuid.uuid4())


def title_from_prompt(prompt: str) -> str:
    words = " ".join(prompt.strip().split()).split()
    return " ".join(words[:6]) + ("..." if len(words) > 6 else "")


def messages_for_thread(thread_id: str) -> list[dict[str, str]]:
    state = get_chatbot(MODE_GENERAL).get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    result = []
    for message in state.values.get("messages", []):
        if isinstance(message, (HumanMessage, AIMessage)) and message.content:
            result.append(
                {
                    "role": "user" if isinstance(message, HumanMessage) else "assistant",
                    "content": str(message.content),
                }
            )
    return result


def select_thread(thread_id: str) -> None:
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = messages_for_thread(thread_id)


def create_thread() -> None:
    thread_id = new_thread_id()
    ensure_thread(thread_id)
    select_thread(thread_id)


if "thread_id" not in st.session_state:
    threads = retrieve_all_threads()
    if threads:
        select_thread(threads[0])
    else:
        create_thread()
if "message_history" not in st.session_state:
    st.session_state["message_history"] = messages_for_thread(st.session_state["thread_id"])
threads = retrieve_all_threads()
current_thread = st.session_state["thread_id"]
if current_thread not in threads:
    ensure_thread(current_thread)
    threads = retrieve_all_threads()

with st.sidebar:
    st.markdown('<div class="brand"><span>✦</span> Cortex</div>', unsafe_allow_html=True)
    if st.button("＋  New chat", use_container_width=True, type="primary"):
        create_thread()
        st.rerun()

    mode = st.selectbox("Workspace mode", MODES, key="workspace_mode")
    if mode == MODE_RAG:
        uploaded_pdf = st.file_uploader("Upload a PDF for this chat", type=["pdf"])
        if uploaded_pdf:
            metadata = thread_document_metadata(current_thread)
            if metadata.get("filename") != uploaded_pdf.name:
                with st.spinner("Indexing PDF..."):
                    ingest_pdf(
                        uploaded_pdf.getvalue(),
                        current_thread,
                        uploaded_pdf.name,
                    )
                st.rerun()
        metadata = thread_document_metadata(current_thread)
        if metadata:
            st.caption(
                f"Indexed: {metadata['filename']} "
                f"({metadata['chunks']} chunks)"
            )
    elif mode == MODE_MCP:
        st.info("MCP tools initialize when you send the first message.")

    st.caption("Your conversations")
    for thread_id in threads:
        title = get_thread_title(thread_id)
        active = thread_id == current_thread
        label = f"•  {title}" if active else title
        if st.button(label, key=f"thread-{thread_id}", use_container_width=True):
            select_thread(thread_id)
            st.rerun()

    st.divider()
    st.caption("SQLite persistence • LangGraph")

if not st.session_state["message_history"]:
    st.markdown(
        '<div class="welcome"><h1>What can Cortex help with?</h1>'
        "<p>Think, create, analyze, and build with your personal AI workspace.</p></div>",
        unsafe_allow_html=True,
    )

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Message Cortex...")
if prompt:
    if get_thread_title(current_thread) == "New chat":
        rename_thread(current_thread, title_from_prompt(prompt))
    st.session_state["message_history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    config = {
        "configurable": {"thread_id": current_thread},
        "metadata": {"mode": mode, "thread_id": current_thread},
        "run_name": f"{mode} chat",
    }
    with st.chat_message("assistant"):
        response = st.write_stream(
            chunk.content
            for chunk, _metadata in stream_chat(
                mode,
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
            )
            if isinstance(chunk, AIMessage) and chunk.content
        )
    st.session_state["message_history"].append({"role": "assistant", "content": response})
    st.rerun()

st.markdown('<div class="disclaimer">Cortex can make mistakes. Verify important information.</div>', unsafe_allow_html=True)
