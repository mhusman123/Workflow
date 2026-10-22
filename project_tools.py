from __future__ import annotations

import re
from html import unescape
from urllib.parse import quote

import requests
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return a short text result."""
    response = requests.get(
        f"https://html.duckduckgo.com/html/?q={quote(query)}",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=15,
    )
    response.raise_for_status()
    text = re.sub(r"<[^>]+>", " ", response.text)
    text = unescape(re.sub(r"\s+", " ", text))
    return text[:5000] or "No search results found."
