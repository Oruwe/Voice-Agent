"""
Knowledge search for agents that call this platform as a tool (REST and MCP).

Moss only queries an index that is loaded into this process; there is no
cloud query. So one client lives for the whole process, each tenant's
knowledge index is loaded once, and every search after that runs in memory.
An upload changes the cloud copy, so refresh() reloads it; searches that
arrive during a reload wait for it instead of failing.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("agent.knowledge")

KNOWLEDGE_LOGICAL_NAME = "knowledge"


def knowledge_index_name(tenant_id: str) -> str:
    return f"{tenant_id}__{KNOWLEDGE_LOGICAL_NAME}"


@dataclass
class KnowledgeHit:
    text: str
    score: float | None
    source: str | None
    chunk: str | None


@dataclass
class KnowledgeResult:
    hits: list[KnowledgeHit] = field(default_factory=list)
    took_ms: float = 0.0
    # False when the tenant has no knowledge index yet (nothing uploaded).
    index_ready: bool = True


class KnowledgeUnavailableError(Exception):
    """Moss could not load or query the index for a reason other than it not existing."""


def _is_not_found(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "index not found" in text or "index_not_found" in text


def _query_options(top_k: int):
    from moss import QueryOptions

    return QueryOptions(top_k=top_k)


def _hit(doc: Any) -> KnowledgeHit:
    metadata = getattr(doc, "metadata", None) or {}
    return KnowledgeHit(
        text=doc.text,
        score=getattr(doc, "score", None),
        source=metadata.get("source"),
        chunk=metadata.get("chunk"),
    )


class KnowledgeService:
    def __init__(self, client_factory: Callable[[], Any]) -> None:
        self._client_factory = client_factory
        self._client: Any = None
        self._loaded: set[str] = set()
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def _lock(self, name: str) -> asyncio.Lock:
        return self._locks.setdefault(name, asyncio.Lock())

    async def _ensure_loaded(self, name: str) -> bool:
        """True once the index is in memory; False if it doesn't exist yet."""
        if name in self._loaded:
            return True
        async with self._lock(name):
            if name in self._loaded:
                return True
            try:
                await self._get_client().load_index(name)
            except Exception as e:
                if _is_not_found(e):
                    return False
                raise KnowledgeUnavailableError(f"could not load {name}") from e
            self._loaded.add(name)
            return True

    async def search(self, *, tenant_id: str, query: str, top_k: int = 3) -> KnowledgeResult:
        name = knowledge_index_name(tenant_id)
        t0 = time.perf_counter()

        def elapsed() -> float:
            return round((time.perf_counter() - t0) * 1000, 1)

        # One retry: a refresh() between the loaded check and the query unloads
        # the index under us, and the second attempt waits for the reload.
        for attempt in (1, 2):
            if not await self._ensure_loaded(name):
                return KnowledgeResult(index_ready=False, took_ms=elapsed())
            try:
                out = await self._get_client().query(name, query, _query_options(top_k))
            except Exception as e:
                self._loaded.discard(name)
                if attempt == 2:
                    raise KnowledgeUnavailableError(f"query failed on {name}") from e
                continue
            hits = [_hit(doc) for doc in (getattr(out, "docs", None) or [])]
            result = KnowledgeResult(hits=hits, took_ms=elapsed())
            logger.info("knowledge query tenant=%s hits=%d took=%.1fms", tenant_id, len(hits), result.took_ms)
            return result
        raise AssertionError("unreachable")

    async def refresh(self, tenant_id: str) -> None:
        """Reload the tenant's index after an upload. Never raises."""
        name = knowledge_index_name(tenant_id)
        async with self._lock(name):
            self._loaded.discard(name)
            client = self._get_client()
            try:
                await client.unload_index(name)
            except Exception:
                pass  # not loaded in this process yet
            try:
                await client.load_index(name)
                self._loaded.add(name)
                logger.info("knowledge index %s reloaded", name)
            except Exception:
                logger.exception("knowledge index %s reload failed; next search retries", name)
