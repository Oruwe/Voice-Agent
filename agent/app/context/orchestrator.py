from __future__ import annotations

import asyncio
import re
from typing import Any

from app.context.moss_provider import MossUnavailableError  # noqa: F401  (re-exported for callers)
from app.context.types import ContextBundle


_DEEP_KNOWLEDGE_TRIGGERS = re.compile(
    r"\b(remember|history|previous|earlier|past|knowledge|document|context)\b",
    re.IGNORECASE,
)

_LIVE_DATA_TRIGGERS = re.compile(
    r"\b(weather|news|stock|price|current|today|latest|live)\b",
    re.IGNORECASE,
)


class LiveOperationalAPI:
    """Placeholder for the real live operational data provider."""

    async def retrieve(
        self,
        *,
        tenant_id: str,
        query_text: str,
    ) -> dict[str, Any]:
        return {
            "status": "unavailable",
            "reason": "Live operational API is not configured",
            "tenant_id": tenant_id,
            "query_text": query_text,
        }


class ContextOrchestrator:
    def __init__(
        self,
        moss: Any,
        qdrant: Any,
        live_api: Any = None,
    ) -> None:
        self._moss = moss
        self._qdrant = qdrant
        self._live_api = live_api

    async def retrieve_context(
        self,
        *,
        tenant_id: str,
        session_id: str,
        query_text: str,
    ) -> ContextBundle:
        bundle = ContextBundle()

        needs_deep = bool(_DEEP_KNOWLEDGE_TRIGGERS.search(query_text)) and self._qdrant is not None
        needs_live = (
            bool(_LIVE_DATA_TRIGGERS.search(query_text))
            and self._live_api is not None
        )

        # Moss (fast) and the optional deep/live sources run concurrently --
        # awaiting Moss first meant every deep/live lookup paid Moss's
        # latency on top of its own, for no reason (they're independent).
        tasks: dict[str, Any] = {
            "moss": self._moss.retrieve_context(
                tenant_id=tenant_id,
                logical_name="session_context",
                query_text=query_text,
            )
        }

        if needs_deep:
            tasks["qdrant"] = self._qdrant.retrieve(
                tenant_id=tenant_id,
                query_text=query_text,
            )

        if needs_live:
            tasks["live"] = self._live_api.retrieve(
                tenant_id=tenant_id,
                query_text=query_text,
            )

        results: list[Any] = await asyncio.gather(*tasks.values(), return_exceptions=True)

        for name, result in zip(tasks.keys(), results):
            if isinstance(result, BaseException):
                if not isinstance(result, Exception):  # e.g. CancelledError
                    raise result
                # MossUnavailableError and any other source failure degrade
                # that one source instead of failing the whole turn.
                bundle.degraded.append(name)
                continue

            bundle.sources_queried.append(name)

            if name == "moss":
                bundle.fast_context = result
            elif name == "qdrant":
                bundle.deep_context = result
            elif name == "live":
                bundle.live_data = result

        return bundle
