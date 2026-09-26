"""
Unit tests for preflight_check.py's LLM chain check.

The real requests need credentials and a network path to each provider,
neither of which exists here -- that is what running preflight on the
deployed worker is for. What is tested is everything around the request:
which providers get checked, with which model, how each is called, and that
a rejected key and an unavailable model come back as different fixes.
"""
import asyncio
import importlib.util
import os
from pathlib import Path
from unittest.mock import patch

import pytest

# preflight_check.py is a top-level operator script, loaded by path like
# latency_report.py in test_latency_report.py.
_PATH = Path(__file__).resolve().parent.parent / "preflight_check.py"
_spec = importlib.util.spec_from_file_location("preflight_check", _PATH)
preflight = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(preflight)


def _env(**overrides):
    base = {"GROQ_API_KEY": "gsk-test", "GOOGLE_API_KEY": "goog-test", "SARVAM_API_KEY": "sv-test"}
    base.update(overrides)
    return patch.dict(os.environ, base, clear=False)


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# llm_request: each provider is called the way the agent's client calls it
# --------------------------------------------------------------------------

def test_groq_request_is_openai_compatible_with_bearer_key():
    host, url, headers, body = preflight.llm_request("groq", "llama-3.3-70b-versatile", "gsk-x")
    assert host == "api.groq.com"
    assert url == "https://api.groq.com/openai/v1/chat/completions"
    assert headers == {"Authorization": "Bearer gsk-x"}
    assert body["model"] == "llama-3.3-70b-versatile"
    assert body["max_tokens"] == 1


def test_gemini_key_goes_in_a_header_never_the_url():
    """A URL can end up in an exception message, and from there in logs."""
    _, url, headers, body = preflight.llm_request("gemini", "gemini-3.5-flash-lite", "secret-key")
    assert "secret-key" not in url
    assert headers == {"x-goog-api-key": "secret-key"}
    assert url.endswith("/models/gemini-3.5-flash-lite:generateContent")
    assert body["generationConfig"]["maxOutputTokens"] == 1


@pytest.mark.parametrize("model, version", [
    ("sarvam-105b-conversations", "v1"),
    ("sarvam-105b", "v2"),
])
def test_sarvam_endpoint_version_follows_the_plugin(model, version):
    _, url, headers, _ = preflight.llm_request("sarvam", model, "sv-x")
    assert url == f"https://api.sarvam.ai/{version}/chat/completions"
    assert headers == {"api-subscription-key": "sv-x"}


# --------------------------------------------------------------------------
# classify_llm_response: the verdict names the fix
# --------------------------------------------------------------------------

def test_200_is_authenticated():
    r = preflight.classify_llm_response("groq", "m", 200, "{}")
    assert r.status == "AUTHENTICATED"


@pytest.mark.parametrize("provider, status, body, key_var", [
    ("groq", 401, '{"error":{"code":"invalid_api_key"}}', "GROQ_API_KEY"),
    ("gemini", 400, '{"error":{"message":"API key not valid. Please pass a valid API key."}}', "GOOGLE_API_KEY"),
    ("sarvam", 403, "forbidden", "SARVAM_API_KEY"),
])
def test_rejected_key_names_the_key_variable(provider, status, body, key_var):
    r = preflight.classify_llm_response(provider, "m", status, body)
    assert r.status == "REJECTED"
    assert key_var in r.detail
    assert "model" not in r.detail


@pytest.mark.parametrize("provider, status, body, model_var", [
    ("groq", 404, '{"error":{"code":"model_not_found"}}', "GROQ_MODEL"),
    ("groq", 400, '{"error":{"code":"model_decommissioned"}}', "GROQ_MODEL"),
    ("gemini", 404, "models/gemini-x is not found for API version v1beta", "GEMINI_MODEL"),
    ("sarvam", 400, "Invalid model", "SARVAM_LLM_MODEL"),
])
def test_unavailable_model_names_the_model_variable(provider, status, body, model_var):
    r = preflight.classify_llm_response(provider, "old-model", status, body)
    assert r.status == "REJECTED"
    assert model_var in r.detail
    assert "old-model" in r.detail


def test_rate_limit_says_the_key_works():
    r = preflight.classify_llm_response("groq", "m", 429, "rate limit")
    assert r.status == "ERROR"
    assert "429" in r.detail and "key works" in r.detail


def test_unexpected_status_is_an_error_with_the_body():
    r = preflight.classify_llm_response("groq", "m", 500, "upstream exploded")
    assert r.status == "ERROR"
    assert "upstream exploded" in r.detail


# --------------------------------------------------------------------------
# check_llm: model and key come from the same place the worker reads them
# --------------------------------------------------------------------------

def test_missing_key_is_reported_without_a_request():
    async def post(*_):
        raise AssertionError("must not send a request without a key")

    with _env(GROQ_API_KEY=""):
        r = _run(preflight.check_llm("groq", post=post))
    assert r.status == "MISSING"
    assert "GROQ_API_KEY" in r.detail


def test_check_uses_the_model_the_worker_will_call():
    sent = {}

    async def post(url, headers, payload):
        sent.update(url=url, payload=payload)
        return 200, "{}"

    with _env(GROQ_MODEL="llama-custom"):
        r = _run(preflight.check_llm("groq", post=post))
    assert r.status == "AUTHENTICATED"
    assert sent["payload"]["model"] == "llama-custom"


def test_transport_failure_is_an_error_not_a_crash():
    async def post(*_):
        raise ConnectionResetError("peer reset")

    with _env():
        r = _run(preflight.check_llm("groq", post=post))
    assert r.status == "ERROR"
    assert "ConnectionResetError" in r.detail


# --------------------------------------------------------------------------
# llm_checks: follows LLM_CHAIN
# --------------------------------------------------------------------------

def _close(checks):
    for _, pending in checks:
        if asyncio.iscoroutine(pending):
            pending.close()


def test_checks_follow_chain_order_and_label_the_model():
    with _env(LLM_CHAIN="sarvam,groq", GROQ_MODEL="llama-x"):
        checks = preflight.llm_checks()
    _close(checks)
    labels = [name for name, _ in checks]
    assert labels == ["LLM sarvam (sarvam-105b-conversations)", "LLM groq (llama-x)"]


def test_default_chain_checks_groq_then_gemini():
    with _env():
        os.environ.pop("LLM_CHAIN", None)
        checks = preflight.llm_checks()
    _close(checks)
    assert [n.split()[1] for n, _ in checks] == ["groq", "gemini"]


def test_invalid_chain_is_one_error_result_not_a_crash():
    with _env(LLM_CHAIN="groq,nope"):
        checks = preflight.llm_checks()
    assert len(checks) == 1
    name, result = checks[0]
    assert result.status == "ERROR"
    assert "nope" in result.detail
