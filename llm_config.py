from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


def get_setting(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st

        return st.secrets.get(name, default)
    except Exception:
        return default


GROQ_API_KEY = get_setting("GROQ_API_KEY")
GROQ_MODEL = get_setting("GROQ_MODEL", DEFAULT_GROQ_MODEL)

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is required. Set it in .env or Streamlit secrets."
    )


def create_groq_llm(**kwargs: Any) -> ChatGroq:
    return ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, **kwargs)
