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


# An onset older than this is not the cause of the stop we're seeing, so
# pairing them would invent a large, wrong barge-in number.
_STALE_ONSET_S = 10.0


def _event_time(ev) -> float:
    # Explicit None check, not `or`: a created_at of 0.0 is falsy and would
    # silently become wall-clock time, pairing two incomparable clocks.
    ts = getattr(ev, "created_at", None)
    return time.time() if ts is None else float(ts)


def _write_row(row: dict) -> None:
    """Append one JSON line. No-op when disabled; never raises."""
    path = os.environ.get("LATENCY_LOG_PATH", "").strip()
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except Exception:
        logger.debug("could not write latency row", exc_info=True)


def record_turn_metrics(item, *, session_id: str) -> None:
    """Append one turn's latency metrics as a JSON line.

    No-op when LATENCY_LOG_PATH is unset. Opens in append mode per call so a
    worker that dies mid-call still leaves a complete, valid file behind.
    """
    if not os.environ.get("LATENCY_LOG_PATH", "").strip():
        return

    try:
        metrics = getattr(item, "metrics", None) or {}
        recorded = {k: metrics[k] for k in _METRIC_FIELDS if k in metrics}
        if not recorded:
            # Turns legitimately carry no timings (a `say()` greeting, an
            # interrupted reply). Writing an empty row would dilute the
            # percentiles with turns that were never measured.
            return

        _write_row({
            "ts": time.time(),
            "session_id": session_id,
            "role": getattr(item, "role", None),
            "item_id": getattr(item, "id", None),
            "interrupted": bool(getattr(item, "interrupted", False)),
            **recorded,
        })
    except Exception:
        logger.debug("could not record turn latency", exc_info=True)


class BargeInTracker:
    """Times how long the agent takes to stop once the user talks over it.

    WHY NOT LiveKit's InterruptionMetrics: that is emitted only by the
    `AdaptiveInterruptionDetector` (inference/interruption.py), and every
    field on it describes model inference. We pin interruption mode to
    "vad", so that detector is never constructed and those metrics never
    fire. The timing has to come from the public state events instead.

    The measurement is (agent leaves "speaking") - (user entered "speaking"
    while the agent was speaking), recorded only when the resulting
    assistant message carries `interrupted=True`. That flag is what
    separates a real barge-in from a "mmhm" the user spoke over the top of
    without ever stopping the agent.

    ORDERING: the assistant item is emitted *before* the agent leaves the
    speaking state (agent_activity.py:3344-3357), so the interrupted flag
    always arrives first and the row is written on the state change.
    """

    def __init__(self, *, session_id: str) -> None:
        self._session_id = session_id
        self._onset: float | None = None
        self._interrupted_pending = False

    def on_user_state(self, ev, *, agent_state: str) -> None:
        if getattr(ev, "new_state", None) == "speaking" and agent_state == "speaking":
            self._onset = _event_time(ev)

    def on_item(self, item) -> None:
        if getattr(item, "role", None) == "assistant" and getattr(item, "interrupted", False):
            self._interrupted_pending = True

    def on_agent_state(self, ev) -> None:
        if getattr(ev, "old_state", None) != "speaking" or getattr(ev, "new_state", None) == "speaking":
            return

        stopped_at = _event_time(ev)
        onset, interrupted = self._onset, self._interrupted_pending
        # Cleared on every exit from "speaking": an overlap that did not stop
        # the agent must not leak into a later turn's measurement.
        self._onset = None
        self._interrupted_pending = False

        if not interrupted or onset is None:
            return
        latency = stopped_at - onset
        if not 0 <= latency <= _STALE_ONSET_S:
            return

        _write_row({
            "ts": time.time(),
            "session_id": self._session_id,
            "role": "barge_in",
            "barge_in_latency": latency,
        })
