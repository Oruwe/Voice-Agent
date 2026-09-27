"""
MCP (Model Context Protocol) server for the knowledge base, stateless
Streamable HTTP: every POST carries one JSON-RPC message (or a batch) and gets
a plain JSON reply. No SSE stream and no session id are needed for a single
read-only tool, which keeps this dependency-free and horizontally trivial.

The transport (auth, parsing, HTTP status) lives in token_service.py; this
module only maps JSON-RPC messages to results, given a search function.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from app.context.knowledge_service import KnowledgeResult, KnowledgeUnavailableError

SUPPORTED_PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "fieldops-knowledge", "title": "FieldOps Knowledge", "version": "1.0.0"}
MAX_TOP_K = 10

SEARCH_TOOL: dict[str, Any] = {
    "name": "search_knowledge",
    "title": "Search the knowledge base",
    "description": (
        "Search this organisation's uploaded manuals, SOPs and policies and return the "
        "most relevant passages with their source document. Call it before answering "
        "any question about procedures, equipment, safety or policy, and answer from "
        "the passages it returns."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The question or topic, in natural language."},
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_TOP_K,
                "default": 3,
                "description": "How many passages to return.",
            },
        },
        "required": ["query"],
    },
    "annotations": {"readOnlyHint": True, "openWorldHint": False},
}

SearchFn = Callable[[str, int], Awaitable[KnowledgeResult]]


def _result(msg_id: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _tool_error(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": True}


def format_hits(result: KnowledgeResult) -> str:
    if not result.index_ready:
        return "No documents have been uploaded to this knowledge base yet."
    if not result.hits:
        return "No matching passages found in the knowledge base."
    blocks = []
    for i, hit in enumerate(result.hits, 1):
        label = hit.source or "document"
        score = f", score {hit.score:.2f}" if hit.score is not None else ""
        blocks.append(f"[{i}] {label}{score}\n{hit.text}")
    return "\n\n".join(blocks)


async def _call_search(arguments: Any, search: SearchFn) -> dict:
    if not isinstance(arguments, dict):
        return _tool_error("Arguments must be an object with a 'query' string.")
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return _tool_error("'query' is required and must be a non-empty string.")
    top_k = arguments.get("top_k", 3)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= MAX_TOP_K:
        return _tool_error(f"'top_k' must be an integer from 1 to {MAX_TOP_K}.")
    try:
        result = await search(query.strip(), top_k)
    except KnowledgeUnavailableError:
        return _tool_error("The knowledge base is temporarily unavailable. Try again shortly.")
    return {
        "content": [{"type": "text", "text": format_hits(result)}],
        "structuredContent": {
            "index_ready": result.index_ready,
            "took_ms": result.took_ms,
            "hits": [
                {"text": h.text, "score": h.score, "source": h.source, "chunk": h.chunk}
                for h in result.hits
            ],
        },
        "isError": False,
    }


async def handle_message(message: Any, *, search: SearchFn) -> dict | None:
    """Return the JSON-RPC response for one message, or None for a notification."""
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
        msg_id = message.get("id") if isinstance(message, dict) else None
        return _error(msg_id, -32600, "Invalid Request")

    method = message["method"]
    if "id" not in message:
        return None  # notifications (e.g. notifications/initialized) get no reply
    msg_id = message["id"]
    params = message.get("params") or {}

    if method == "initialize":
        requested = params.get("protocolVersion") if isinstance(params, dict) else None
        version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else SUPPORTED_PROTOCOL_VERSIONS[0]
        return _result(msg_id, {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
            "instructions": (
                "Use search_knowledge to look up this organisation's procedures, equipment "
                "manuals, safety rules and policies before answering questions about them."
            ),
        })
    if method == "ping":
        return _result(msg_id, {})
    if method == "tools/list":
        return _result(msg_id, {"tools": [SEARCH_TOOL]})
    if method == "tools/call":
        if not isinstance(params, dict) or params.get("name") != SEARCH_TOOL["name"]:
            name = params.get("name") if isinstance(params, dict) else None
            return _error(msg_id, -32602, f"Unknown tool: {name}")
        return _result(msg_id, await _call_search(params.get("arguments", {}), search))
    return _error(msg_id, -32601, f"Method not found: {method}")
