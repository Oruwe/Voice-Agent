"""
The knowledge tool other agents call: KnowledgeService (Moss boundary mocked),
POST /v1/knowledge/query, POST /v1/agent-keys, the upload size cap, and the
MCP endpoint. Auth is patched out except where auth itself is under test.
"""
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

import app.security.token_service as token_service
from app.context.knowledge_service import (
    KnowledgeHit,
    KnowledgeResult,
    KnowledgeService,
    KnowledgeUnavailableError,
)
from app.security.jwt_auth import decode_access_token

AUTH = {"Authorization": "Bearer test-token"}


def _doc(text, score, source="sop.pdf", chunk="0"):
    return SimpleNamespace(id=f"{source}:{chunk}", text=text, score=score, metadata={"source": source, "chunk": chunk})


def _moss(docs=(), load_error=None):
    client = MagicMock()
    client.load_index = AsyncMock(side_effect=load_error)
    client.unload_index = AsyncMock()
    client.query = AsyncMock(return_value=SimpleNamespace(docs=list(docs), time_taken_ms=1.1))
    return client


# --- KnowledgeService -------------------------------------------------------

@pytest.mark.asyncio
async def test_index_loads_once_then_every_search_runs_in_memory():
    client = _moss([_doc("Seal P-7 every 2000 h", 0.91)])
    svc = KnowledgeService(lambda: client)

    first = await svc.search(tenant_id="acme", query="pump seal", top_k=3)
    second = await svc.search(tenant_id="acme", query="pump seal", top_k=3)

    client.load_index.assert_awaited_once_with("acme__knowledge")
    assert client.query.await_count == 2
    assert first.hits == [KnowledgeHit(text="Seal P-7 every 2000 h", score=0.91, source="sop.pdf", chunk="0")]
    assert second.index_ready


@pytest.mark.asyncio
async def test_tenant_without_uploads_gets_an_empty_ready_false_result_not_an_error():
    client = _moss(load_error=RuntimeError("Cloud error: Index not found: acme__knowledge (INDEX_NOT_FOUND)"))
    svc = KnowledgeService(lambda: client)

    result = await svc.search(tenant_id="acme", query="anything")

    assert result.index_ready is False
    assert result.hits == []
    client.query.assert_not_called()


@pytest.mark.asyncio
async def test_a_real_load_failure_is_surfaced():
    svc = KnowledgeService(lambda: _moss(load_error=RuntimeError("HTTP 429 credit_exhausted")))
    with pytest.raises(KnowledgeUnavailableError):
        await svc.search(tenant_id="acme", query="anything")


@pytest.mark.asyncio
async def test_query_failure_reloads_and_retries_once_before_giving_up():
    client = _moss([_doc("ok", 0.5)])
    client.query = AsyncMock(side_effect=[RuntimeError("Index not loaded"), SimpleNamespace(docs=[_doc("ok", 0.5)])])
    svc = KnowledgeService(lambda: client)

    result = await svc.search(tenant_id="acme", query="q")

    assert [h.text for h in result.hits] == ["ok"]
    assert client.load_index.await_count == 2

    client.query = AsyncMock(side_effect=RuntimeError("still broken"))
    with pytest.raises(KnowledgeUnavailableError):
        await svc.search(tenant_id="acme", query="q")


@pytest.mark.asyncio
async def test_refresh_reloads_so_a_new_upload_is_searchable_and_never_raises():
    client = _moss()
    svc = KnowledgeService(lambda: client)
    await svc.search(tenant_id="acme", query="q")

    await svc.refresh("acme")
    client.unload_index.assert_awaited_once_with("acme__knowledge")
    assert client.load_index.await_count == 2

    client.load_index = AsyncMock(side_effect=RuntimeError("boom"))
    await svc.refresh("acme")  # logged, not raised


# --- HTTP endpoints -------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MOSS_PROJECT_ID", "proj")
    monkeypatch.setenv("MOSS_PROJECT_KEY", "key")
    monkeypatch.setenv("JWT_SECRET", "x" * 40)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    identity = SimpleNamespace(tenant_slug="acme", external_id="tech-1", tenant_id="t-uuid")
    with patch.object(token_service, "_resolve_bearer", AsyncMock(return_value=identity)):
        yield TestClient(token_service.app)


def _knowledge(result=None, error=None):
    svc = MagicMock()
    svc.search = AsyncMock(return_value=result, side_effect=error)
    svc.refresh = AsyncMock()
    return patch.object(token_service, "_knowledge", svc)


HIT_RESULT = KnowledgeResult(
    hits=[KnowledgeHit(text="Lock out the breaker first.", score=0.88, source="safety.pdf", chunk="2")],
    took_ms=11.8,
)


def test_query_returns_hits_scoped_to_the_callers_tenant(client):
    with _knowledge(HIT_RESULT) as svc:
        resp = client.post("/v1/knowledge/query", headers=AUTH, json={"query": "how to isolate", "top_k": 2})

    assert resp.status_code == 200
    body = resp.json()
    assert body["hits"][0] == {"text": "Lock out the breaker first.", "score": 0.88, "source": "safety.pdf", "chunk": "2"}
    assert body["took_ms"] == 11.8 and body["index_ready"] is True
    svc.search.assert_awaited_once_with(tenant_id="acme", query="how to isolate", top_k=2)


def test_query_validates_input(client):
    with _knowledge(HIT_RESULT):
        assert client.post("/v1/knowledge/query", headers=AUTH, json={"query": ""}).status_code == 422
        assert client.post("/v1/knowledge/query", headers=AUTH, json={"query": "x", "top_k": 50}).status_code == 422


def test_query_reports_moss_outage_as_502(client):
    with _knowledge(error=KnowledgeUnavailableError("down")):
        resp = client.post("/v1/knowledge/query", headers=AUTH, json={"query": "x"})
    assert resp.status_code == 502


def test_agent_key_is_a_long_lived_token_for_the_same_identity(client):
    resp = client.post("/v1/agent-keys", headers=AUTH)

    assert resp.status_code == 200
    body = resp.json()
    payload = decode_access_token(body["agent_key"])
    assert payload.sub == "tech-1" and payload.tenant_slug == "acme"
    assert body["expires_in"] == 30 * 86400
    assert payload.exp - int(time.time()) > 29 * 86400


def test_upload_over_the_cap_is_rejected_before_parsing(client, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.001")  # ~1 KB
    with patch.object(token_service, "parse_document") as parse:
        resp = client.post(
            "/v1/documents/upload", headers=AUTH, files={"file": ("big.txt", b"x" * 5000, "text/plain")},
        )
    assert resp.status_code == 413
    assert "big.txt" in resp.json()["detail"]
    parse.assert_not_called()


def test_successful_upload_reloads_the_search_index(client):
    provider = MagicMock()
    provider.upsert_context = AsyncMock()
    with _knowledge(HIT_RESULT) as svc, patch.object(token_service, "MossContextProvider", return_value=provider):
        resp = client.post(
            "/v1/documents/upload", headers=AUTH,
            files={"file": ("sop.txt", b"Pump P-7 needs a new seal every 2000 hours. " * 30, "text/plain")},
        )
    assert resp.status_code == 200
    svc.refresh.assert_awaited_once_with("acme")


# --- MCP ---------------------------------------------------------------

def _rpc(client, method, params=None, msg_id=1):
    msg = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        msg["params"] = params
    return client.post("/mcp", headers=AUTH, json=msg)


def test_mcp_initialize_negotiates_the_clients_version(client):
    body = _rpc(client, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}).json()
    assert body["result"]["protocolVersion"] == "2025-06-18"
    assert body["result"]["capabilities"] == {"tools": {"listChanged": False}}

    unknown = _rpc(client, "initialize", {"protocolVersion": "1999-01-01"}).json()
    assert unknown["result"]["protocolVersion"] == "2025-11-25"


def test_mcp_lists_the_search_tool(client):
    tools = _rpc(client, "tools/list").json()["result"]["tools"]
    assert [t["name"] for t in tools] == ["search_knowledge"]
    assert tools[0]["inputSchema"]["required"] == ["query"]


def test_mcp_tool_call_returns_text_and_structured_hits(client):
    with _knowledge(HIT_RESULT) as svc:
        result = _rpc(client, "tools/call", {"name": "search_knowledge", "arguments": {"query": "isolate", "top_k": 1}}).json()["result"]

    assert result["isError"] is False
    assert "safety.pdf" in result["content"][0]["text"] and "Lock out the breaker" in result["content"][0]["text"]
    assert result["structuredContent"]["hits"][0]["source"] == "safety.pdf"
    svc.search.assert_awaited_once_with(tenant_id="acme", query="isolate", top_k=1)


def test_mcp_tool_errors_are_results_not_protocol_errors(client):
    with _knowledge(error=KnowledgeUnavailableError("down")):
        down = _rpc(client, "tools/call", {"name": "search_knowledge", "arguments": {"query": "x"}}).json()["result"]
    assert down["isError"] is True

    with _knowledge(HIT_RESULT) as svc:
        bad = _rpc(client, "tools/call", {"name": "search_knowledge", "arguments": {"top_k": 3}}).json()["result"]
        svc.search.assert_not_called()
    assert bad["isError"] is True and "query" in bad["content"][0]["text"]


def test_mcp_protocol_errors(client):
    assert _rpc(client, "tools/call", {"name": "rm_rf", "arguments": {}}).json()["error"]["code"] == -32602
    assert _rpc(client, "resources/list").json()["error"]["code"] == -32601
    assert _rpc(client, "ping").json()["result"] == {}

    note = client.post("/mcp", headers=AUTH, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert note.status_code == 202 and note.content == b""

    parse = client.post("/mcp", headers={**AUTH, "Content-Type": "application/json"}, content=b"{not json")
    assert parse.status_code == 400 and parse.json()["error"]["code"] == -32700

    batch = client.post("/mcp", headers=AUTH, json=[
        {"jsonrpc": "2.0", "id": 1, "method": "ping"},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
    ]).json()
    assert batch == [{"jsonrpc": "2.0", "id": 1, "result": {}}]

    assert client.get("/mcp").status_code == 405


def test_mcp_requires_a_bearer_token():
    resp = TestClient(token_service.app).post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
    assert resp.status_code == 401
