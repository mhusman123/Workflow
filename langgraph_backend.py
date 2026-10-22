from __future__ import annotations

import asyncio
import base64
import logging
import os
import queue
import sqlite3
import sys
import tempfile
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Annotated, Any, Optional, TypedDict

import requests
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from llm_config import create_groq_llm
from local_embeddings import LocalHashEmbeddings
from project_tools import web_search

load_dotenv()

MODE_GENERAL = "General Chat"
MODE_TOOLS = "Tools"
MODE_RAG = "PDF / RAG"
MODE_MCP = "MCP"
MODES = (MODE_GENERAL, MODE_TOOLS, MODE_RAG, MODE_MCP)
MAX_ATTACHMENT_TEXT = 30000
MAX_IMAGE_BYTES = 4_000_000
SUPPORTED_ATTACHMENT_TYPES = (
    "png",
    "jpg",
    "jpeg",
    "webp",
    "gif",
    "pdf",
    "txt",
    "md",
    "csv",
    "json",
    "py",
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_message_content(
    prompt: str, attachments: list[dict[str, Any]]
) -> str | list[dict[str, Any]]:
    """Build LangChain content blocks for text files and images."""
    text_parts = [prompt.strip()] if prompt.strip() else []
    image_blocks: list[dict[str, Any]] = []

    for attachment in attachments:
        name = str(attachment.get("name", "attachment"))
        mime_type = str(attachment.get("mime_type") or "application/octet-stream")
        data = attachment.get("data", b"")
        if not isinstance(data, bytes):
            raise TypeError(f"Attachment data for {name} must be bytes.")

        if mime_type.startswith("image/"):
            if len(data) > MAX_IMAGE_BYTES:
                try:
                    from io import BytesIO
                    from PIL import Image

                    image = Image.open(BytesIO(data)).convert("RGB")
                    image.thumbnail((1600, 1600))
                    output = BytesIO()
                    image.save(output, format="JPEG", quality=85, optimize=True)
                    data = output.getvalue()
                    mime_type = "image/jpeg"
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not resize image '{name}': {exc}"
                    ) from exc
            encoded = base64.b64encode(data).decode("ascii")
            image_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded}",
                    },
                }
            )
            text_parts.append(f"[Attached image: {name}]")
            continue

        if mime_type == "application/pdf" or name.lower().endswith(".pdf"):
            try:
                from pypdf import PdfReader
                from io import BytesIO

                pages = PdfReader(BytesIO(data)).pages
                extracted = "\n\n".join(
                    page.extract_text() or "" for page in pages
                ).strip()
            except Exception as exc:
                raise RuntimeError(f"Could not read PDF '{name}': {exc}") from exc
        elif mime_type.startswith("text/") or name.lower().endswith(
            (".txt", ".md", ".csv", ".json", ".py")
        ):
            extracted = data.decode("utf-8", errors="replace")
        else:
            raise ValueError(
                f"Unsupported file type for '{name}'. "
                "Use an image, PDF, or text-based file."
            )

        text_parts.append(
            f"[Attached file: {name}]\n{extracted[:MAX_ATTACHMENT_TEXT]}"
        )

    if not image_blocks:
        return "\n\n".join(text_parts)[:MAX_ATTACHMENT_TEXT]
    return [{"type": "text", "text": "\n\n".join(text_parts)}, *image_blocks]


llm = create_groq_llm()
embeddings = LocalHashEmbeddings()
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
conn.execute(
    """
    CREATE TABLE IF NOT EXISTS cortex_threads (
        thread_id TEXT PRIMARY KEY,
        title TEXT NOT NULL DEFAULT 'New chat',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted_at TEXT
    )
    """
)
conn.commit()
checkpointer = SqliteSaver(conn=conn)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_thread(thread_id: str, title: str = "New chat") -> None:
    timestamp = _now()
    conn.execute(
        """
        INSERT INTO cortex_threads (thread_id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(thread_id) DO NOTHING
        """,
        (thread_id, title, timestamp, timestamp),
    )
    conn.commit()


def retrieve_all_threads() -> list[str]:
    checkpoint_thread_ids = set()
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config["configurable"].get("thread_id")
        if thread_id:
            checkpoint_thread_ids.add(str(thread_id))
    for thread_id in checkpoint_thread_ids:
        ensure_thread(thread_id)

    rows = conn.execute(
        """
        SELECT thread_id FROM cortex_threads
        WHERE deleted_at IS NULL
        ORDER BY updated_at DESC
        """
    ).fetchall()
    return [row[0] for row in rows]


def get_thread_title(thread_id: str) -> str:
    row = conn.execute(
        "SELECT title FROM cortex_threads WHERE thread_id = ? AND deleted_at IS NULL",
        (thread_id,),
    ).fetchone()
    return row[0] if row else "New chat"


def rename_thread(thread_id: str, title: str) -> None:
    clean_title = " ".join(title.strip().split())
    if not clean_title:
        raise ValueError("Chat name cannot be empty.")
    ensure_thread(thread_id)
    conn.execute(
        "UPDATE cortex_threads SET title = ?, updated_at = ? WHERE thread_id = ?",
        (clean_title[:80], _now(), thread_id),
    )
    conn.commit()


def delete_thread(thread_id: str) -> None:
    ensure_thread(thread_id)
    timestamp = _now()
    conn.execute(
        "UPDATE cortex_threads SET deleted_at = ?, updated_at = ? WHERE thread_id = ?",
        (timestamp, timestamp, thread_id),
    )
    conn.commit()


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform add, sub, mul, or div on two numbers."""
    if operation == "add":
        result = first_num + second_num
    elif operation == "sub":
        result = first_num - second_num
    elif operation == "mul":
        result = first_num * second_num
    elif operation == "div":
        if second_num == 0:
            return {"error": "Division by zero is not allowed"}
        result = first_num / second_num
    else:
        return {"error": f"Unsupported operation '{operation}'"}
    return {
        "first_num": first_num,
        "second_num": second_num,
        "operation": operation,
        "result": result,
    }


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch the latest stock quote for a symbol."""
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return {"error": "Set ALPHAVANTAGE_API_KEY to use stock prices."}
    response = requests.get(
        "https://www.alphavantage.co/query",
        params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": api_key},
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


_THREAD_RETRIEVERS: dict[str, tuple[list[str], list[list[float]]]] = {}
_THREAD_METADATA: dict[str, dict[str, Any]] = {}


def _split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    step = max(1, chunk_size - overlap)
    return [text[start : start + chunk_size] for start in range(0, len(text), step)]


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """Index an uploaded PDF in memory for one conversation."""
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
    try:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Install pypdf to index PDF files.") from exc
        pages = PdfReader(temp_path).pages
        chunks = [
            chunk
            for page in pages
            for chunk in _split_text(page.extract_text() or "")
            if chunk
        ]
        _THREAD_RETRIEVERS[str(thread_id)] = (
            chunks,
            embeddings.embed_documents(chunks),
        )
        summary = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(pages),
            "chunks": len(chunks),
        }
        _THREAD_METADATA[str(thread_id)] = summary
        return summary
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def thread_document_metadata(thread_id: str) -> dict[str, Any]:
    return _THREAD_METADATA.get(str(thread_id), {})


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """Retrieve relevant passages from the PDF uploaded to this conversation."""
    retriever = _THREAD_RETRIEVERS.get(str(thread_id)) if thread_id else None
    if retriever is None:
        return {"error": "No document indexed for this chat. Upload a PDF first."}
    chunks, vectors = retriever
    query_vector = embeddings.embed_query(query)
    ranked = [
        (sum(left * right for left, right in zip(query_vector, vector)), chunk)
        for chunk, vector in zip(chunks, vectors)
    ]
    context = [chunk for _, chunk in sorted(ranked, reverse=True)[:4]]
    return {
        "query": query,
        "context": context,
        "source_file": thread_document_metadata(str(thread_id)).get("filename"),
    }


def _build_graph(tools: list[BaseTool], rag: bool = False):
    model = llm.bind_tools(tools) if tools else llm
    vision_model = create_groq_llm(
        model=os.getenv(
            "GROQ_VISION_MODEL",
            "qwen/qwen3.8-27b",
        ),
        max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "600")),
    )
    if tools:
        vision_model = vision_model.bind_tools(tools)

    def chat_node(state: ChatState, config: dict | None = None):
        messages = state["messages"]
        system_content = (
            "Answer concisely and accurately. Prefer a direct answer or a few "
            "short bullets. Do not repeat the full attached file or question."
        )
        if rag:
            thread_id = (config or {}).get("configurable", {}).get("thread_id")
            system_content += (
                " Use rag_tool for questions about the uploaded PDF and always "
                f"pass thread_id={thread_id!r}."
            )
        messages = [SystemMessage(content=system_content), *messages]
        has_image = any(
            isinstance(message.content, list)
            and any(
                isinstance(block, dict) and block.get("type") == "image_url"
                for block in message.content
            )
            for message in messages
        )
        active_model = vision_model if has_image else model
        return {"messages": [active_model.invoke(messages, config=config)]}

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    if tools:
        graph.add_node("tools", ToolNode(tools))
        graph.add_conditional_edges("chat_node", tools_condition)
        graph.add_edge("tools", "chat_node")
    else:
        graph.add_edge("chat_node", END)
    return graph.compile(checkpointer=checkpointer)


_graphs: dict[str, Any] = {}


def _get_sync_graph(mode: str):
    if mode not in (MODE_GENERAL, MODE_TOOLS, MODE_RAG):
        raise ValueError(f"{mode} is not a synchronous mode.")
    if mode not in _graphs:
        tools: list[BaseTool] = []
        if mode in (MODE_TOOLS, MODE_RAG):
            tools = [web_search, get_stock_price, calculator]
        if mode == MODE_RAG:
            tools.append(rag_tool)
        _graphs[mode] = _build_graph(tools, rag=mode == MODE_RAG)
    return _graphs[mode]


def _mcp_loop_worker() -> None:
    asyncio.set_event_loop(_mcp_loop)
    _mcp_loop.run_forever()


_mcp_loop = asyncio.new_event_loop()
_mcp_thread = threading.Thread(target=_mcp_loop_worker, daemon=True)
_mcp_thread.start()
_mcp_graph: Any = None
_mcp_future: Any = None


async def _create_mcp_graph():
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as exc:
        raise RuntimeError(
            "Install langchain-mcp-adapters to use MCP mode."
        ) from exc

    servers = {
        "expense": {
            "transport": "streamable_http",
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp",
        }
    }
    math_server = os.getenv("MCP_MATH_SERVER_PATH")
    if math_server:
        servers["arith"] = {
            "transport": "stdio",
            "command": sys.executable,
            "args": [math_server],
        }
    client = MultiServerMCPClient(servers)
    mcp_tools = await client.get_tools()
    return _build_graph([web_search, get_stock_price, *mcp_tools])


def _get_mcp_graph():
    global _mcp_graph, _mcp_future
    if _mcp_graph is not None:
        return _mcp_graph
    if _mcp_future is None:
        _mcp_future = asyncio.run_coroutine_threadsafe(
            _create_mcp_graph(), _mcp_loop
        )
    try:
        _mcp_graph = _mcp_future.result(timeout=30)
    except Exception as exc:
        _mcp_future = None
        raise RuntimeError(f"Unable to initialize MCP mode: {exc}") from exc
    return _mcp_graph


def get_chatbot(mode: str):
    """Return the graph for a selected UI mode, initializing optional MCP lazily."""
    if mode == MODE_MCP:
        return _get_mcp_graph()
    return _get_sync_graph(mode)


def stream_chat(
    mode: str, messages: dict[str, list[BaseMessage]], config: dict
) -> Iterator[tuple[BaseMessage, dict]]:
    """Stream all modes through one frontend-compatible interface."""
    if mode != MODE_MCP:
        yield from get_chatbot(mode).stream(
            messages, config=config, stream_mode="messages"
        )
        return

    events: queue.Queue[Any] = queue.Queue()

    async def run() -> None:
        try:
            async for item in _get_mcp_graph().astream(
                messages, config=config, stream_mode="messages"
            ):
                events.put(item)
        except Exception as exc:
            events.put(exc)
        finally:
            events.put(None)

    asyncio.run_coroutine_threadsafe(run(), _mcp_loop)
    while True:
        item = events.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise RuntimeError(f"MCP chat failed: {item}") from item
        yield item
