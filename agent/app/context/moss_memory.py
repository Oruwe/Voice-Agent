"""
Zero-latency conversational memory on Moss.

Why this exists
---------------
Making the LLM *decide* to call a `retrieve_context` tool costs a full extra
LLM round trip (hundreds of ms) before the agent can say a word -- Moss's
sub-10ms search was hidden behind that cost. This module puts Moss directly
inline in the agent's `llm_node` instead: every reply is grounded in a local
Moss query that runs in single-digit milliseconds, with no tool hop.

Two Moss indexes, both queried in-process (no network on the hot path):

* **Live memory** -- `client.session("{tenant}__session_context")`. A local
  `SessionIndex`: `add_docs` embeds locally (Rust core) and `query` runs
  in-memory. If the cloud index already exists it is auto-loaded, so the agent
  remembers earlier calls for the same tenant. Every user/agent turn is added
  as it happens. At shutdown `push_index()` persists it back to Moss cloud.
* **Knowledge base** -- `client.load_index("{tenant}__knowledge")`. An index
  built ahead of time (manuals, SOPs, site info). Optional: if it doesn't
  exist the agent just runs on live memory.

Everything here is best-effort: a Moss failure degrades to "no extra context",
it never blocks or breaks a voice turn.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("agent.moss_memory")


@dataclass
class RecallHit:
    id: str
    text: str
    score: float | None
    source: str  # "memory" | "knowledge"


@dataclass
class RecallResult:
    hits: list[RecallHit] = field(default_factory=list)
    elapsed_ms: float = 0.0
    moss_ms: float | None = None  # Moss's own reported search time
    timed_out: bool = False

    def as_prompt(self) -> str:
        memory = [h.text for h in self.hits if h.source == "memory"]
        knowledge = [h.text for h in self.hits if h.source == "knowledge"]
        parts = []
        if knowledge:
            parts.append("Knowledge base:\n- " + "\n- ".join(knowledge))
        if memory:
            parts.append("Earlier in conversation(s) with this user:\n- " + "\n- ".join(memory))
        return "\n\n".join(parts)


def _doc_info(**kwargs: Any):
    # Imported lazily so unit tests can run without the native moss_core wheel.
    from moss import DocumentInfo

    return DocumentInfo(**kwargs)


def _query_options(**kwargs: Any):
    from moss import QueryOptions

    return QueryOptions(**kwargs)


class MossLiveMemory:
    def __init__(
        self,
        client: Any,
        *,
        tenant_id: str,
        session_id: str,
        top_k: int = 3,
        min_score: float = 0.0,
        budget_ms: float = 60.0,
        memory_logical_name: str = "session_context",
        knowledge_logical_name: str = "knowledge",
    ) -> None:
        self._client = client
        self._tenant_id = tenant_id
        self._session_id = session_id
        self._top_k = top_k
        self._min_score = min_score
        self._budget_s = budget_ms / 1000.0
        self._memory_name = f"{tenant_id}__{memory_logical_name}"
        self._knowledge_name = f"{tenant_id}__{knowledge_logical_name}"

        self._session: Any = None
        self._knowledge_loaded = False
        self._ready = asyncio.Event()
        self._write_lock = asyncio.Lock()
        self._turn = 0

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    async def start(self) -> None:
        """Open the local session index and preload the knowledge index, in
        parallel. Call this as a background task at job start -- recall()
        simply returns nothing until it finishes, so it never delays the
        greeting."""
        t0 = time.perf_counter()

        async def _open_session() -> None:
            try:
                self._session = await self._client.session(self._memory_name)
            except Exception:
                logger.exception("moss: could not open session index %s", self._memory_name)

        async def _load_knowledge() -> None:
            try:
                await self._client.load_index(self._knowledge_name)
                self._knowledge_loaded = True
            except Exception as e:  # index simply may not exist for this tenant
                logger.info("moss: knowledge index %s not loaded (%s)", self._knowledge_name, e)

        await asyncio.gather(_open_session(), _load_knowledge())
        self._ready.set()
        logger.info(
            "moss ready in %.0f ms (memory=%s, knowledge=%s)",
            (time.perf_counter() - t0) * 1000,
            self._session is not None,
            self._knowledge_loaded,
        )

    async def recall(self, query: str, *, exclude_ids: set[str] | None = None) -> RecallResult:
        """Query live memory + knowledge concurrently within a hard latency
        budget. Never raises."""
        t0 = time.perf_counter()
        result = RecallResult()
        query = (query or "").strip()
        if not self.ready or len(query) < 3:
            return result

        exclude_ids = exclude_ids or set()
        # Ask for a couple extra so excluding the current turn still leaves top_k.
        opts = _query_options(top_k=self._top_k + len(exclude_ids))

        jobs: dict[str, Any] = {}
        if self._session is not None:
            jobs["memory"] = self._session.query(query, opts)
        if self._knowledge_loaded:
            jobs["knowledge"] = self._client.query(self._knowledge_name, query, opts)
        if not jobs:
            return result

        try:
            outs = await asyncio.wait_for(
                asyncio.gather(*jobs.values(), return_exceptions=True), timeout=self._budget_s
            )
        except asyncio.TimeoutError:
            result.timed_out = True
            result.elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.warning("moss recall exceeded %.0f ms budget", self._budget_s * 1000)
            return result

        moss_times: list[float] = []
        for source, out in zip(jobs.keys(), outs):
            if isinstance(out, Exception):
                logger.warning("moss %s query failed: %s", source, out)
                continue
            taken = getattr(out, "time_taken_ms", None)
            if taken is not None:
                moss_times.append(float(taken))
            for doc in (getattr(out, "docs", None) or []):
                doc_id = str(getattr(doc, "id", ""))
                score = getattr(doc, "score", None)
                if doc_id in exclude_ids:
                    continue
                if score is not None and score < self._min_score:
                    continue
                result.hits.append(RecallHit(id=doc_id, text=doc.text, score=score, source=source))

        result.hits.sort(key=lambda h: (h.score is None, -(h.score or 0.0)))
        result.hits = result.hits[: self._top_k]
        result.moss_ms = max(moss_times) if moss_times else None
        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        return result

    async def remember(self, *, role: str, text: str, doc_id: str | None = None) -> None:
        """Add one conversation turn to the local session index. Local
        embedding, no network. Never raises."""
        text = (text or "").strip()
        if self._session is None or not text:
            return
        async with self._write_lock:
            self._turn += 1
            doc = _doc_info(
                id=doc_id or f"{self._session_id}:{self._turn}",
                text=f"{role}: {text}",
                metadata={"role": role, "session_id": self._session_id, "ts": str(int(time.time()))},
            )
            try:
                await self._session.add_docs([doc])
            except Exception:
                logger.exception("moss: failed to add turn to session index")

    async def close(self, *, timeout: float = 10.0) -> None:
        """Persist the live memory to Moss cloud so the next call remembers it."""
        if self._session is None or getattr(self._session, "doc_count", 0) == 0:
            return
        try:
            await asyncio.wait_for(self._session.push_index(), timeout=timeout)
            logger.info("moss: pushed %s docs to %s", self._session.doc_count, self._memory_name)
        except Exception:
            logger.exception("moss: push_index failed")
