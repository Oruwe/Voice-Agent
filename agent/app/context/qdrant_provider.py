"""
Qdrant provider for the deep knowledge retrieval path.

Sits in ContextOrchestrator's `qdrant` slot. Queried only when the user's
message matches _DEEP_KNOWLEDGE_TRIGGERS (history, document, SOP, etc.) —
so it never sits on the hot path for simple operational queries.

Uses qdrant-client's FastEmbed integration: embedding happens in-process
(Rust/ONNX, no API call), so the first query downloads the model (~80 MB)
but every subsequent one is local.

Collection naming mirrors Moss: `{tenant_slug}__knowledge`.
If a tenant's collection doesn't exist yet Qdrant returns an empty list —
the agent degrades gracefully rather than erroring.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("agent.qdrant")

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim, ~80 MB, fast


@dataclass
class QdrantHit:
    id: str
    text: str
    score: float


class QdrantProvider:
    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        top_k: int = 3,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self._url = url or os.environ.get("QDRANT_URL", "http://localhost:6333")
        self._api_key = api_key or os.environ.get("QDRANT_API_KEY") or None
        self._top_k = top_k
        self._model = model
        self._client: Any = None

    def _get_client(self):
        if self._client is None:
            from qdrant_client import AsyncQdrantClient

            self._client = AsyncQdrantClient(url=self._url, api_key=self._api_key)
            # Pin the fastembed model so upsert (client.add) and query
            # (client.query query_text=) always use the same embedder.
            self._client.set_model(self._model)
        return self._client

    @staticmethod
    def _collection(tenant_id: str) -> str:
        return f"{tenant_id}__knowledge"

    async def retrieve(self, *, tenant_id: str, query_text: str) -> list[QdrantHit]:
        """Search the tenant knowledge collection. Returns [] if the
        collection doesn't exist or Qdrant is unavailable."""
        from qdrant_client.http.exceptions import UnexpectedResponse

        client = self._get_client()
        collection = self._collection(tenant_id)
        try:
            results = await client.query(
                collection_name=collection,
                query_text=query_text,
                limit=self._top_k,
                with_payload=True,
            )
        except UnexpectedResponse as e:
            if e.status_code == 404:
                logger.debug("qdrant: collection %s not found for tenant %s", collection, tenant_id)
                return []
            logger.warning("qdrant query failed for %s: %s", collection, e)
            return []
        except Exception as e:
            logger.warning("qdrant query failed for %s: %s", collection, e)
            return []

        hits: list[QdrantHit] = []
        for point in results:
            payload = point.payload or {}
            text = payload.get("text") or payload.get("content") or str(payload)
            hits.append(QdrantHit(id=str(point.id), text=text, score=point.score))
        return hits

    async def upsert(self, *, tenant_id: str, docs: list[dict]) -> None:
        """Index documents into the tenant knowledge collection.
        Each doc must have 'id' and 'text' keys; 'metadata' is optional.

        Uses client.add() so the same fastembed model handles both indexing
        and querying — consistent with client.query(query_text=...) in retrieve().
        Collection is auto-created if it doesn't exist.
        """
        client = self._get_client()
        collection = self._collection(tenant_id)

        texts = [doc["text"] for doc in docs]
        metadatas = [{"text": doc["text"], **(doc.get("metadata") or {})} for doc in docs]
        ids = [
            doc["id"] if isinstance(doc["id"], int)
            else abs(hash(str(doc["id"]))) % (2**63)
            for doc in docs
        ]

        await client.add(
            collection_name=collection,
            documents=texts,
            metadata=metadatas,
            ids=ids,
        )
