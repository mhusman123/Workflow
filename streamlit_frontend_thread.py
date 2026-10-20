import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from langgraph_backend import chatbot

# Set page configuration
st.set_page_config(
    page_title="Cortex",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Cortex Aesthetic
st.markdown("""
<style>
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    .main .block-container {
        max-width: 840px;
        padding-top: 1.5rem;
        padding-bottom: 6rem;
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.15);
    }

    /* ---- Chat Title Buttons ---- */
    div.chat-btn-container div.stButton > button {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        background: transparent !important;
        width: 100% !important;
        text-align: left !important;
        padding: 0.48rem 0.6rem !important;
        border-radius: 8px !important;
        color: #ececec !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        transition: background-color 0.15s ease;
    }
    div.chat-btn-container div.stButton > button:hover {
        background-color: rgba(128,128,128,0.12) !important;
    }
    div.chat-btn-container div.stButton > button:focus,
    div.chat-btn-container div.stButton > button:focus-visible,
    div.chat-btn-container div.stButton > button:active {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* ---- Chat Message Bubbles ---- */
    .stChatMessage {
        background-color: transparent !important;
        padding: 1.25rem 0 !important;
        border-bottom: 1px solid rgba(128,128,128,0.08);
    }
    .stChatMessage [data-testid="stChatMessageContent"] {
        font-size: 1.02rem;
        line-height: 1.7;
    }
    .hero-container { text-align: center; margin-top: 3.5rem; margin-bottom: 2.5rem; }
    .hero-title { font-size: 2.2rem; font-weight: 600; margin-bottom: 0.5rem; }
    .hero-subtitle { font-size: 1.05rem; opacity: 0.75; margin-bottom: 2rem; }
    .footer-disclaimer { text-align: center; font-size: 0.78rem; opacity: 0.6; margin-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ---- Utility Functions ----

def make_chat_title(text):
    cleaned = " ".join((text or "").strip().split())
    if not cleaned: return "New chat"
    words = cleaned.split(" ")
    return " ".join(words[:4]) + ("..." if len(words) > 4 else "")

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    tid = generate_thread_id()
    st.session_state.update({
        "thread_id": tid,
        "message_history": [],
        "renaming_thread_id": None,
    })
    add_thread(tid)
    st.session_state["thread_titles"][tid] = "New chat"

def add_thread(tid):
    if tid not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(tid)

def ensure_thread_title(tid):
    if tid in st.session_state["thread_titles"]:
        return st.session_state["thread_titles"][tid]
    try:
        msgs = load_conversation(tid)
    except Exception:
        msgs = []
    first = next((m.content for m in msgs if isinstance(m, HumanMessage) and m.content), "")
    title = make_chat_title(first)
    st.session_state["thread_titles"][tid] = title
    return title

def load_conversation(tid):
    state = chatbot.get_state(config={"configurable": {"thread_id": tid}})
    return state.values.get("messages", [])

def delete_thread(tid):
    st.session_state["chat_threads"] = [t for t in st.session_state["chat_threads"] if t != tid]
    st.session_state["thread_titles"].pop(tid, None)
    st.session_state.pop("renaming_thread_id", None)
    if st.session_state["thread_id"] == tid:
        if st.session_state["chat_threads"]:
            switch_to_thread(st.session_state["chat_threads"][-1])
        else:
            reset_chat()

def switch_to_thread(tid):
    st.session_state["thread_id"] = tid
    st.session_state["renaming_thread_id"] = None
    msgs = load_conversation(tid)
    st.session_state["message_history"] = [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in msgs
    ]

# ---- Session Init ----
for key, default in [
    ("message_history", []),
    ("thread_id", generate_thread_id()),
    ("chat_threads", []),
    ("thread_titles", {}),
    ("pending_prompt", None),
    ("renaming_thread_id", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

add_thread(st.session_state["thread_id"])
ensure_thread_title(st.session_state["thread_id"])

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### 🤖 **Cortex**")
    if st.button("New chat", use_container_width=True, type="primary"):
        reset_chat()
        st.rerun()

    st.markdown("---")
    st.caption("Recent Conversations")

    for tid in st.session_state["chat_threads"][::-1]:
        title = ensure_thread_title(tid)
        is_active = (tid == st.session_state["thread_id"])

        # Inline Rename Editor
        if st.session_state.get("renaming_thread_id") == tid:
            c1, c2, c3 = st.columns([0.66, 0.17, 0.17])
            with c1:
                new_name = st.text_input("", value=title, key=f"ren-inp-{tid}", label_visibility="collapsed")
            with c2:
                if st.button("✓", key=f"ren-save-{tid}"):
                    if new_name.strip():
                        st.session_state["thread_titles"][tid] = new_name.strip()
                    st.session_state["renaming_thread_id"] = None
                    st.rerun()
            with c3:
                if st.button("✕", key=f"ren-cancel-{tid}"):
                    st.session_state["renaming_thread_id"] = None
                    st.rerun()
        else:
            # Clean chat button — no dots
            st.markdown(f'<div class="chat-btn-container">', unsafe_allow_html=True)
            if st.button(title, key=f"chat-btn-{tid}", use_container_width=True):
                if is_active:
                    # Toggle action row on click of active chat
                    if st.session_state.get("action_open_tid") == tid:
                        st.session_state["action_open_tid"] = None
                    else:
                        st.session_state["action_open_tid"] = tid
                else:
                    switch_to_thread(tid)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

            # Action row for active chat
            if st.session_state.get("action_open_tid") == tid:
                ac1, ac2 = st.columns(2)
                with ac1:
                    if st.button("✏️ Rename", key=f"act-ren-{tid}", use_container_width=True):
                        st.session_state["renaming_thread_id"] = tid
                        st.session_state["action_open_tid"] = None
                        st.rerun()
                with ac2:
                    if st.button("🗑️ Delete", key=f"act-del-{tid}", use_container_width=True):
                        delete_thread(tid)
                        st.session_state["action_open_tid"] = None
                        st.rerun()

    st.markdown("---")
    st.caption("Powered by LangGraph")

# ---- Empty State / Suggestions ----
if len(st.session_state["message_history"]) == 0:
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">What can Cortex help with today?</div>
        <div class="hero-subtitle">Ask questions, brainstorm ideas, analyze logic, or code together.</div>
    </div>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💡 **Brainstorm concepts**\n\nFresh angles for a modern AI application", use_container_width=True):
            st.session_state["pending_prompt"] = "Give me 5 creative ideas for a modern AI software application."
            st.rerun()
        if st.button("📝 **Summarize & Explain**\n\nExplain quantum computing simply", use_container_width=True):
            st.session_state["pending_prompt"] = "Explain quantum computing in simple terms for a beginner."
            st.rerun()
    with c2:
        if st.button("💻 **Write Python Code**\n\nBuild a REST API using FastAPI", use_container_width=True):
            st.session_state["pending_prompt"] = "Write a complete example of a REST API using FastAPI with CRUD endpoints."
            st.rerun()
        if st.button("🎯 **Debug & Refactor**\n\nCore practices for clean backend architecture", use_container_width=True):
            st.session_state["pending_prompt"] = "What are the core best practices for clean backend code architecture?"
            st.rerun()

# ---- Message Feed ----
for message in st.session_state["message_history"]:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# ---- Chat Input ----
prompt_from_card = st.session_state.pop("pending_prompt", None)
user_input = st.chat_input("Message Cortex...") or prompt_from_card

if user_input:
    current_title = st.session_state["thread_titles"].get(st.session_state["thread_id"])
    if current_title in (None, "New chat"):
        st.session_state["thread_titles"][st.session_state["thread_id"]] = make_chat_title(user_input)

    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}
    with st.chat_message("assistant", avatar="🤖"):
        def ai_only_stream():
            for chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(chunk, AIMessage):
                    yield chunk.content
        ai_message = st.write_stream(ai_only_stream())

    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
    st.rerun()

st.markdown('<div class="footer-disclaimer">Cortex can make mistakes. Verify important information.</div>', unsafe_allow_html=True)
