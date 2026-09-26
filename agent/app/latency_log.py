"""
Per-turn latency recorder: appends LiveKit's own `ChatMessage.metrics` to a
JSONL file so `latency_report.py` can aggregate real p50/p95 numbers.

WHY THIS EXISTS: every latency claim about this agent was previously
"it feels faster". LiveKit already measures each turn end to end
(`livekit.agents.llm.chat_context.MetricsReport`) -- `e2e_latency`,
`llm_node_ttft` (which includes our inline Moss recall, since Moss runs
inside `llm_node`), `tts_node_ttfb`, and `end_of_turn_delay` (what
MIN_ENDPOINTING_DELAY moves). Nothing persisted it, so nothing could be
measured across a call. This does, and nothing more.

OFF BY DEFAULT: does nothing at all unless LATENCY_LOG_PATH is set, so
production pays neither the disk write nor the open().

NEVER RECORDS TRANSCRIPT TEXT: only timings, roles, and ids. A benchmark
artifact has no business holding conversation content -- this project
isolates tenant data (see docs/adr/001-audit-tenant-id.md) and a latency
file is not an audited store.

NEVER RAISES: a telemetry failure must never break a live call.
"""
from __future__ import annotations

import json
import logging
import os
import time

logger = logging.getLogger("agent.latency")

# Only these keys are read off MetricsReport. Anything LiveKit adds later is
# ignored rather than blindly copied, which is what keeps transcript text
# (and any future free-text field) out of the file by construction.
_METRIC_FIELDS = (
    "e2e_latency",
    "llm_node_ttft",
    "llm_node_ttfs",
    "llm_node_tps",
    "tts_node_ttfb",
    "playback_latency",
    "end_of_turn_delay",
    "transcription_delay",
    "on_user_turn_completed_delay",
)


def record_turn_metrics(item, *, session_id: str) -> None:
    """Append one turn's latency metrics as a JSON line.

    No-op when LATENCY_LOG_PATH is unset. Opens in append mode per call so a
    worker that dies mid-call still leaves a complete, valid file behind.
    """
    path = os.environ.get("LATENCY_LOG_PATH", "").strip()
    if not path:
        return

    try:
        metrics = getattr(item, "metrics", None) or {}
        recorded = {k: metrics[k] for k in _METRIC_FIELDS if k in metrics}
        if not recorded:
            # Turns legitimately carry no timings (a `say()` greeting, an
            # interrupted reply). Writing an empty row would dilute the
            # percentiles with turns that were never measured.
            return

        row = {
            "ts": time.time(),
            "session_id": session_id,
            "role": getattr(item, "role", None),
            "item_id": getattr(item, "id", None),
            **recorded,
        }
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except Exception:
        logger.debug("could not record turn latency", exc_info=True)
