"""Tests for the zero-latency Moss path: MossLiveMemory and the agent's
llm_node context injection. Moss itself is mocked at the client boundary."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from livekit.agents.llm import ChatContext, ChatMessage

from app.agent_entrypoint import FieldOpsAssistant
from app.context.moss_memory import MossLiveMemory, RecallHit, RecallResult
from app.context.moss_provider import MossUnavailableError
from app.context.orchestrator import ContextOrchestrator


def _doc(id, text, score):
    return SimpleNamespace(id=id, text=text, score=score, metadata=None)


def _search(docs, ms=2.5):
    return SimpleNamespace(docs=docs, time_taken_ms=ms)


def make_client(session_docs=(), knowledge_docs=(), knowledge_exists=True):
    session = MagicMock()
    session.query = AsyncMock(return_value=_search(list(session_docs), 1.2))
    session.add_docs = AsyncMock(return_value=(1, 0))
    session.push_index = AsyncMock()
    session.doc_count = 2
    client = MagicMock()
    client.session = AsyncMock(return_value=session)
    client.load_index = AsyncMock(
        side_effect=None if knowledge_exists else RuntimeError("index not found")
    )
    client.query = AsyncMock(return_value=_search(list(knowledge_docs), 3.4))
    return client, session


@pytest.mark.asyncio
async def test_recall_before_ready_returns_nothing_without_querying():
    client, session = make_client()
    mem = MossLiveMemory(client, tenant_id="t", session_id="s")
    result = await mem.recall("where is the transformer")
    assert result.hits == []
    session.query.assert_not_called()


@pytest.mark.asyncio
async def test_start_opens_tenant_scoped_indexes():
    client, _ = make_client()
    mem = MossLiveMemory(client, tenant_id="acme", session_id="room1")
    await mem.start()
    assert mem.ready
    client.session.assert_awaited_once_with("acme__session_context")
    client.load_index.assert_awaited_once_with("acme__knowledge")


@pytest.mark.asyncio
async def test_missing_knowledge_index_still_ready_with_memory_only():
    client, session = make_client(session_docs=[_doc("1", "user: pump 4 leaking", 0.9)], knowledge_exists=False)
    mem = MossLiveMemory(client, tenant_id="t", session_id="s")
    await mem.start()
    result = await mem.recall("pump status")
    assert [h.source for h in result.hits] == ["memory"]
    client.query.assert_not_called()


@pytest.mark.asyncio
async def test_recall_merges_sources_sorts_by_score_and_excludes_ids():
    client, _ = make_client(
        session_docs=[_doc("cur", "user: pump status", 0.99), _doc("m1", "user: pump 4 leaking", 0.7)],
        knowledge_docs=[_doc("k1", "Pump 4 SOP: close valve B first", 0.8)],
    )
    mem = MossLiveMemory(client, tenant_id="t", session_id="s", top_k=2)
    await mem.start()
    result = await mem.recall("pump status", exclude_ids={"cur"})
    assert [h.id for h in result.hits] == ["k1", "m1"]
    assert result.moss_ms == 3.4
    assert not result.timed_out
    prompt = result.as_prompt()
    assert "Knowledge base" in prompt and "close valve B" in prompt and "pump 4 leaking" in prompt


@pytest.mark.asyncio
async def test_recall_respects_latency_budget():
    client, session = make_client()

    async def slow(*a, **k):
        await asyncio.sleep(0.5)
        return _search([])

    session.query = slow
    client.query = slow
    mem = MossLiveMemory(client, tenant_id="t", session_id="s", budget_ms=20)
    await mem.start()
    result = await mem.recall("anything at all")
    assert result.timed_out and result.hits == []
    assert result.elapsed_ms < 200


@pytest.mark.asyncio
async def test_recall_survives_query_failure():
    client, session = make_client(knowledge_docs=[_doc("k1", "SOP", 0.5)])
    session.query = AsyncMock(side_effect=RuntimeError("boom"))
    mem = MossLiveMemory(client, tenant_id="t", session_id="s")
    await mem.start()
    result = await mem.recall("sop please")
    assert [h.id for h in result.hits] == ["k1"]


@pytest.mark.asyncio
async def test_remember_adds_role_prefixed_doc_and_close_pushes():
    client, session = make_client()
    mem = MossLiveMemory(client, tenant_id="t", session_id="s")
    await mem.start()
    await mem.remember(role="user", text="  pump 4 is leaking ", doc_id="msg_1")
    doc = session.add_docs.await_args.args[0][0]
    assert doc.id == "msg_1" and doc.text == "user: pump 4 is leaking"
    await mem.remember(role="user", text="   ")  # empty -> ignored
    assert session.add_docs.await_count == 1
    await mem.close()
    session.push_index.assert_awaited_once()


# --------------------------------------------------------------------------
# FieldOpsAssistant.llm_node injection
# --------------------------------------------------------------------------

def _assistant(memory):
    return FieldOpsAssistant(
        tenant_id="t", session_id="s", orchestrator=AsyncMock(), memory=memory,
    )


@pytest.mark.asyncio
async def test_llm_node_injects_moss_context_before_last_user_message():
    memory = MagicMock()
    memory.recall = AsyncMock(return_value=RecallResult(
        hits=[RecallHit(id="k1", text="Pump 4 SOP: close valve B", score=0.9, source="knowledge")],
        elapsed_ms=3.0, moss_ms=2.0,
    ))
    assistant = _assistant(memory)
    ctx = ChatContext()
    ctx.add_message(role="assistant", content="Hi")
    user = ctx.add_message(role="user", content="how do I fix pump 4")

    seen = {}

    async def fake_default(agent, chat_ctx, tools, model_settings):
        seen["ctx"] = chat_ctx
        yield "ok"

    with patch("app.agent_entrypoint.Agent.default.llm_node", fake_default):
        out = [c async for c in assistant.llm_node(ctx, [], None)]

    assert out == ["ok"]
    memory.recall.assert_awaited_once_with("how do I fix pump 4", exclude_ids={user.id})
    roles = [i.role for i in seen["ctx"].items]
    assert roles == ["assistant", "system", "user"]
    assert "close valve B" in seen["ctx"].items[1].text_content
    assert len(ctx.items) == 2  # original context untouched


@pytest.mark.asyncio
async def test_llm_node_reuses_recall_on_tool_loop_reentry_and_skips_empty():
    memory = MagicMock()
    memory.recall = AsyncMock(return_value=RecallResult())
    assistant = _assistant(memory)
    ctx = ChatContext()
    ctx.add_message(role="user", content="status of ticket")

    async def fake_default(agent, chat_ctx, tools, model_settings):
        assert [i.role for i in chat_ctx.items] == ["user"]  # nothing injected
        yield "x"

    with patch("app.agent_entrypoint.Agent.default.llm_node", fake_default):
        [c async for c in assistant.llm_node(ctx, [], None)]
        [c async for c in assistant.llm_node(ctx, [], None)]
    assert memory.recall.await_count == 1


@pytest.mark.asyncio
async def test_llm_node_without_memory_passes_through():
    assistant = _assistant(None)
    ctx = ChatContext()
    ctx.add_message(role="user", content="hello")

    async def fake_default(agent, chat_ctx, tools, model_settings):
        yield "y"

    with patch("app.agent_entrypoint.Agent.default.llm_node", fake_default):
        assert [c async for c in assistant.llm_node(ctx, [], None)] == ["y"]


# --------------------------------------------------------------------------
# Orchestrator: sources run concurrently, failures degrade
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_runs_moss_and_qdrant_concurrently():
    order = []

    async def moss_q(**kw):
        order.append("moss-start")
        await asyncio.sleep(0.05)
        order.append("moss-end")
        return [{"text": "fast"}]

    async def qdrant_q(**kw):
        order.append("qdrant-start")
        await asyncio.sleep(0.01)
        return ["deep"]

    moss = MagicMock(retrieve_context=moss_q)
    qdrant = MagicMock(retrieve=qdrant_q)
    bundle = await ContextOrchestrator(moss, qdrant).retrieve_context(
        tenant_id="t", session_id="s", query_text="what is the history here"
    )
    assert order.index("qdrant-start") < order.index("moss-end")
    assert bundle.fast_context == [{"text": "fast"}] and bundle.deep_context == ["deep"]


@pytest.mark.asyncio
async def test_orchestrator_degrades_moss_and_tolerates_no_qdrant():
    moss = MagicMock(retrieve_context=AsyncMock(side_effect=MossUnavailableError("down")))
    bundle = await ContextOrchestrator(moss, None).retrieve_context(
        tenant_id="t", session_id="s", query_text="history please"
    )
    assert bundle.degraded == ["moss"]
    assert bundle.fast_context == []
