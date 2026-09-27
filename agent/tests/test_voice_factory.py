"""
Unit tests for app.voice_providers.factory: the latency-relevant options the
Sarvam STT and TTS are built with.

The real plugin classes are constructed (no request is sent until a stream
opens), so a renamed option in the pinned plugin fails here, not on a call.
"""
import os
from unittest.mock import patch

import pytest

from app.voice_providers import factory


@pytest.fixture(autouse=True)
def _key():
    with patch.dict(os.environ, {"SARVAM_API_KEY": "sv-test"}):
        yield


def test_stt_asks_sarvam_to_act_on_flush_by_default(monkeypatch):
    """Without flush_signal=true the server ignores the plugin's flush, and
    the final transcript that starts the LLM waits on the server's timeout."""
    monkeypatch.delenv("SARVAM_STT_FLUSH", raising=False)
    assert factory.build_stt()._opts.flush_signal is True


def test_stt_flush_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("SARVAM_STT_FLUSH", "false")
    assert factory.build_stt()._opts.flush_signal is False


def test_tts_tokenizer_thresholds_default_low(monkeypatch):
    for var in ("TTS_MIN_SENTENCE_LEN", "TTS_SENTENCE_LOOKAHEAD"):
        monkeypatch.delenv(var, raising=False)
    config = factory.build_tts()._opts.word_tokenizer._config
    assert (config.min_sentence_len, config.stream_context_len) == (8, 3)


def test_pinned_tts_plugin_still_reads_the_tokenizer_we_set():
    """build_tts sets a private attribute (sarvam.TTS has no tokenizer
    argument). If an upgrade renames it, fail here instead of silently
    falling back to the slow default."""
    from dataclasses import fields

    from livekit.plugins import sarvam

    tts = factory.build_tts()
    assert isinstance(tts, sarvam.TTS)
    assert "word_tokenizer" in {f.name for f in fields(tts._opts)}


def _first_token(tokenizer, chunks):
    import asyncio

    async def run():
        stream = tokenizer.stream()
        for chunk in chunks:
            stream.push_text(chunk)
        try:
            return await asyncio.wait_for(stream.__anext__(), timeout=0.2)
        except asyncio.TimeoutError:
            return None
        finally:
            await stream.aclose()

    return asyncio.run(run())


# A reply mid-generation: the first sentence is complete, the second isn't.
_MID_REPLY = ("It is ", "offline. ", "A technician is on the ")


def test_short_first_sentence_is_sent_while_the_reply_is_still_generating():
    token = _first_token(factory.first_sentence_tokenizer(), _MID_REPLY)
    assert token is not None and token.token == "It is offline."


def test_plugin_default_tokenizer_would_have_held_it():
    """The regression this guards: with the plugin's default, the same
    sentence isn't sent until the next one is complete too."""
    from livekit.agents import tokenize

    assert _first_token(tokenize.basic.SentenceTokenizer(), _MID_REPLY) is None
