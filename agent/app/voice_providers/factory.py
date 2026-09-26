"""
Builds the STT/TTS pair for the agent session using the official
`livekit-plugins-sarvam` plugin. (The LLM chain lives next door in
llm_chain.py.)

Why the official plugin, not a hand-rolled port:
* STT keeps ONE streaming WebSocket open for the whole call (no per-utterance
  TLS handshake), returns interim transcripts (which is what lets LiveKit's
  preemptive generation start the LLM before the user finishes speaking),
  uses Sarvam's own server-side VAD, and resamples room audio to 16 kHz
  correctly.
* TTS keeps a pooled, prewarmable WebSocket, so the first audio byte of a
  reply doesn't wait on a fresh connection.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("agent.voice")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


def build_stt():
    from livekit.plugins import sarvam

    return sarvam.STT(
        api_key=os.environ["SARVAM_API_KEY"],
        # "unknown" = auto-detect per utterance, so a worker can switch
        # between English/Hindi/Kannada mid-call.
        language=_env("SARVAM_STT_LANGUAGE", "unknown"),
        model=_env("SARVAM_STT_MODEL", "saaras:v3"),
        # "codemix" is worth trying for Hinglish/Kanglish speakers.
        mode=_env("SARVAM_STT_MODE", "transcribe"),
        high_vad_sensitivity=_env("SARVAM_STT_HIGH_VAD", "true").lower() == "true",
    )


def build_tts():
    from livekit.plugins import sarvam

    return sarvam.TTS(
        api_key=os.environ["SARVAM_API_KEY"],
        target_language_code=_env("SARVAM_TTS_LANGUAGE", "en-IN"),
        model=_env("SARVAM_TTS_MODEL", "bulbul:v3"),
        speaker=_env("SARVAM_TTS_SPEAKER", "shubh"),
        output_audio_codec="linear16",
        speech_sample_rate=int(_env("SARVAM_TTS_SAMPLE_RATE", "24000")),
        min_buffer_size=int(_env("SARVAM_TTS_MIN_BUFFER", "30")),
        max_chunk_length=int(_env("SARVAM_TTS_MAX_CHUNK", "150")),
        pace=float(_env("SARVAM_TTS_PACE", "1.05")),
    )


def prewarm_tts(tts) -> None:
    """Open the TTS WebSocket before the first reply is needed."""
    prewarm = getattr(tts, "prewarm", None)
    if callable(prewarm):
        try:
            prewarm()
        except Exception:
            logger.exception("tts prewarm failed (non-fatal)")
