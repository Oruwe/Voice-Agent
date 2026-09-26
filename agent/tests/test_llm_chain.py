"""
Unit tests for app.voice_providers.llm_chain: parsing LLM_CHAIN and
assembling the provider list.

The provider builders themselves are patched out -- constructing a real
openai/google/sarvam client needs credentials and reaches the network,
neither of which exists here (see preflight_check.py). What is tested is
the part that decides WHICH providers get built, in what order, and that a
misconfiguration fails at startup rather than mid-call.
"""
import os
from unittest.mock import patch

import pytest

from app.voice_providers import llm_chain


def _env(**overrides):
    """Environment with all provider keys present unless overridden."""
    base = {
        "GROQ_API_KEY": "gsk-test",
        "GOOGLE_API_KEY": "goog-test",
        "SARVAM_API_KEY": "sarvam-test",
    }
    base.update(overrides)
    return patch.dict(os.environ, base, clear=False)


# --------------------------------------------------------------------------
# parse_chain
# --------------------------------------------------------------------------

def test_default_chain_is_unchanged_from_before_this_module():
    assert llm_chain.parse_chain(llm_chain.DEFAULT_CHAIN) == ["groq", "gemini"]


def test_parse_chain_preserves_order():
    assert llm_chain.parse_chain("sarvam,groq,gemini") == ["sarvam", "groq", "gemini"]


def test_parse_chain_tolerates_whitespace_and_case():
    assert llm_chain.parse_chain(" Sarvam , GROQ ") == ["sarvam", "groq"]


def test_parse_chain_accepts_single_provider():
    assert llm_chain.parse_chain("sarvam") == ["sarvam"]


def test_parse_chain_rejects_unknown_provider_by_name():
    with pytest.raises(ValueError, match="openai"):
        llm_chain.parse_chain("groq,openai")


def test_parse_chain_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        llm_chain.parse_chain("  , ,")


def test_parse_chain_rejects_duplicates():
    """A repeated provider means a typo, not a real second fallback."""
    with pytest.raises(ValueError, match="repeats"):
        llm_chain.parse_chain("groq,gemini,groq")


# --------------------------------------------------------------------------
# build_llm
# --------------------------------------------------------------------------

def test_build_uses_default_chain_when_env_unset():
    with _env():
        os.environ.pop("LLM_CHAIN", None)
        with patch.object(llm_chain, "_build_groq", return_value="GROQ") as groq, \
             patch.object(llm_chain, "_build_gemini", return_value="GEMINI") as gem, \
             patch.object(llm_chain, "_build_sarvam") as sarvam, \
             patch("livekit.agents.llm.FallbackAdapter"):
            llm_chain.build_llm()

    groq.assert_called_once()
    gem.assert_called_once()
    sarvam.assert_not_called()


def test_build_respects_chain_order():
    with _env(LLM_CHAIN="sarvam,groq"):
        with patch.object(llm_chain, "_build_sarvam", return_value="S"), \
             patch.object(llm_chain, "_build_groq", return_value="G"), \
             patch("livekit.agents.llm.FallbackAdapter") as adapter:
            llm_chain.build_llm()
    assert adapter.call_args.args[0] == ["S", "G"]


def test_single_provider_returns_bare_llm_not_a_fallback_adapter():
    from livekit.agents.llm import FallbackAdapter

    with _env(LLM_CHAIN="sarvam"):
        with patch.object(llm_chain, "_build_sarvam", return_value="SARVAM_LLM"):
            result = llm_chain.build_llm()
    assert result == "SARVAM_LLM"
    assert not isinstance(result, FallbackAdapter)


def test_multiple_providers_are_wrapped_in_a_fallback_adapter():
    with _env(LLM_CHAIN="groq,gemini"):
        # FallbackAdapter is imported inside build_llm, so patch it at source.
        with patch.object(llm_chain, "_build_groq", return_value="A"), \
             patch.object(llm_chain, "_build_gemini", return_value="B"), \
             patch("livekit.agents.llm.FallbackAdapter") as adapter:
            llm_chain.build_llm()

    adapter.assert_called_once()
    assert adapter.call_args.args[0] == ["A", "B"]


def test_missing_api_key_fails_at_startup_naming_the_variable():
    """A silently dropped fallback is worse than a refused start."""
    with _env(LLM_CHAIN="sarvam,groq", SARVAM_API_KEY=""):
        with pytest.raises(ValueError, match="SARVAM_API_KEY"):
            llm_chain.build_llm()


def test_missing_key_for_a_provider_not_in_the_chain_is_fine():
    with _env(LLM_CHAIN="gemini", SARVAM_API_KEY="", GROQ_API_KEY=""):
        with patch.object(llm_chain, "_build_gemini", return_value="GEMINI"):
            assert llm_chain.build_llm() == "GEMINI"


def test_groq_sends_reasoning_effort_only_when_configured():
    """gpt-oss reasons before its first answer token; Groq rejects the
    parameter for models that don't reason, so it's never sent by default."""
    with _env(GROQ_REASONING_EFFORT="low"), patch("livekit.plugins.openai.LLM") as llm:
        llm_chain._build_groq()
    assert llm.call_args.kwargs["reasoning_effort"] == "low"

    with _env(), patch("livekit.plugins.openai.LLM") as llm:
        os.environ.pop("GROQ_REASONING_EFFORT", None)
        llm_chain._build_groq()
    assert "reasoning_effort" not in llm.call_args.kwargs


def test_preload_imports_exactly_the_chains_plugins():
    with _env(LLM_CHAIN="groq,sarvam"), patch("importlib.import_module") as imp:
        llm_chain.preload_llm_plugins()
    assert [c.args[0] for c in imp.call_args_list] == ["livekit.plugins.openai", "livekit.plugins.sarvam"]


def test_preload_never_raises():
    """It runs in prewarm: a broken plugin must not take the worker down."""
    with _env(LLM_CHAIN="groq,nope"), patch("importlib.import_module") as imp:
        llm_chain.preload_llm_plugins()  # invalid chain -> no-op
    imp.assert_not_called()

    with _env(LLM_CHAIN="gemini"), patch("importlib.import_module", side_effect=ImportError("boom")):
        llm_chain.preload_llm_plugins()  # must not raise


def test_unknown_provider_in_env_fails_at_startup():
    with _env(LLM_CHAIN="groq,nope"):
        with pytest.raises(ValueError, match="nope"):
            llm_chain.build_llm()
