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
        # The plugin sends {"type": "flush"} the moment Sarvam's VAD hears the
        # user stop, but the server only acts on it with flush_signal=true in
        # the socket URL. Without it the final transcript -- which is what
        # starts the LLM -- waits on the server's own timeout.
        flush_signal=_env("SARVAM_STT_FLUSH", "true").lower() == "true",
    )


def tts_voice() -> tuple[str, str, str]:
    """(model, speaker, language) the agent speaks with. preflight_check.py
    verifies exactly this voice, so the two can't drift apart."""
    return (
        _env("SARVAM_TTS_MODEL", "bulbul:v3"),
        _env("SARVAM_TTS_SPEAKER", "shubh"),
        _env("SARVAM_TTS_LANGUAGE", "en-IN"),
    )


def first_sentence_tokenizer():
    """Splits LLM text into the chunks sent to Sarvam. A sentence is sent once
    the next one starts; the plugin's default also held back any sentence
    under 20 characters, so a short opener like "It is offline." waited for
    the whole following sentence before synthesis could begin."""
    from livekit.agents import tokenize

    return tokenize.basic.SentenceTokenizer(
        min_sentence_len=int(_env("TTS_MIN_SENTENCE_LEN", "8")),
        stream_context_len=int(_env("TTS_SENTENCE_LOOKAHEAD", "3")),
    )


def build_tts():
    from livekit.plugins import sarvam

    model, speaker, language = tts_voice()
    tts = sarvam.TTS(
        api_key=os.environ["SARVAM_API_KEY"],
        target_language_code=language,
        model=model,
        speaker=speaker,
        output_audio_codec="linear16",
        speech_sample_rate=int(_env("SARVAM_TTS_SAMPLE_RATE", "24000")),
        min_buffer_size=int(_env("SARVAM_TTS_MIN_BUFFER", "30")),
        max_chunk_length=int(_env("SARVAM_TTS_MAX_CHUNK", "150")),
        pace=float(_env("SARVAM_TTS_PACE", "1.05")),
    )
    # sarvam.TTS takes no tokenizer argument; livekit-plugins-sarvam is pinned
    # (requirements.txt) and a test asserts this attribute still exists.
    tts._opts.word_tokenizer = first_sentence_tokenizer()
    return tts


def prewarm_tts(tts) -> None:
    """Open the TTS WebSocket before the first reply is needed."""
    prewarm = getattr(tts, "prewarm", None)
    if callable(prewarm):
        try:
            prewarm()
        except Exception:
            logger.exception("tts prewarm failed (non-fatal)")
