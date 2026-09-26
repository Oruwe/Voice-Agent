"""
Builds the LLM fallback chain from the LLM_CHAIN env var.

WHY THIS IS CONFIGURABLE: the chain used to be hardcoded Groq -> Gemini.
On Groq's free tier that degrades badly in practice: `FallbackAdapter` is a
circuit breaker, so one 429 marks Groq unavailable and every subsequent
turn goes to Gemini until a background probe recovers it -- then it 429s
and flips again. You end up running a Gemini agent with a latency spike on
each oscillation, while believing you're on Groq.

WHY SARVAM IS WORTH TRYING: `sarvam.LLM` subclasses the same `OpenAILLM`
the Groq entry already uses (tool calling goes through an identical code
path), it reuses the SARVAM_API_KEY this agent already holds for STT/TTS,
and it runs on Indian infrastructure -- which for an India-based deployment
should dominate every millisecond of local tuning elsewhere in this repo.

Unverified: whether sarvam-105b handles this agent's tool calls as well as
llama-3.3-70b, and whether Sarvam meters LLM usage against the same
rate-limit bucket as STT/TTS. Neither is reachable from the dev sandbox
(see preflight_check.py). The default chain is therefore UNCHANGED --
opt in with LLM_CHAIN and compare with latency_report.py.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("agent.llm")

DEFAULT_CHAIN = "groq,gemini"

# Each provider names the env var it cannot start without, so a typo in
# LLM_CHAIN fails at startup with a usable message instead of a bare
# KeyError mid-call.
_REQUIRED_KEY = {
    "groq": "GROQ_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "sarvam": "SARVAM_API_KEY",
}


# Env var naming each provider's model, and the model used when it's unset.
# preflight_check.py reads this same table, so the model it verifies at boot is
# exactly the one the worker will call.
_MODEL = {
    # llama-3.3-70b-versatile was decommissioned for free/developer keys on
    # 2026-08-16; gpt-oss-120b is Groq's recommended replacement.
    "groq": ("GROQ_MODEL", "openai/gpt-oss-120b"),
    # gemini-2.0/2.5 models refuse new API keys ("no longer available to new
    # users", checked 2026-09-26); 3.5-flash-lite is Google's named successor.
    "gemini": ("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    "sarvam": ("SARVAM_LLM_MODEL", "sarvam-105b-conversations"),
}


def model_for(name: str) -> str:
    """The model the agent will call for provider `name`."""
    env, default = _MODEL[name]
    return os.environ.get(env, default)


def model_env(name: str) -> str:
    """The env var that selects provider `name`'s model."""
    return _MODEL[name][0]


def key_env(name: str) -> str:
    """The env var holding provider `name`'s API key."""
    return _REQUIRED_KEY[name]


# Groq models that take reasoning_effort=low|medium|high. Every other Groq
# model rejects that value, so it is sent only to these.
_GROQ_REASONING_PREFIXES = ("openai/gpt-oss",)


def groq_reasoning_effort(model: str | None = None) -> str | None:
    """GROQ_REASONING_EFFORT for `model` (default: the configured Groq model),
    or None when unset or when the model doesn't reason (then the parameter
    isn't sent).

    gpt-oss models reason before their first answer token, and the voice
    pipeline waits for that token -- "low" keeps the wait short. A model that
    doesn't reason skips the wait entirely, which is why it can be switched to
    with GROQ_MODEL alone, without also clearing this variable.
    """
    effort = os.environ.get("GROQ_REASONING_EFFORT", "").strip() or None
    model = model or model_for("groq")
    if effort and model.startswith(_GROQ_REASONING_PREFIXES):
        return effort
    return None


def _temperature() -> float:
    return float(os.environ.get("LLM_TEMPERATURE", "0.6"))


def _build_groq():
    from livekit.plugins import openai

    extra = {}
    effort = groq_reasoning_effort()
    if effort:
        extra["reasoning_effort"] = effort
    return openai.LLM(
        model=model_for("groq"),
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
        _strict_tool_schema=False,
        temperature=_temperature(),
        **extra,
    )


def _build_gemini():
    from livekit.plugins import google

    return google.LLM(model=model_for("gemini"))


def _build_sarvam():
    from livekit.plugins import sarvam

    return sarvam.LLM(
        # "-conversations" is the multi-turn optimized variant, which is what
        # a voice agent actually is.
        model=model_for("sarvam"),
        api_key=os.environ["SARVAM_API_KEY"],
        temperature=_temperature(),
    )


_PLUGIN_MODULE = {
    "groq": "livekit.plugins.openai",
    "gemini": "livekit.plugins.google",
    "sarvam": "livekit.plugins.sarvam",
}


def preload_llm_plugins() -> None:
    """Import the plugin for each provider in LLM_CHAIN, on the main thread.

    Call at module load of the worker entrypoint: LiveKit only registers
    plugins on the main thread (a call on a worker thread can't import one),
    and importing livekit.plugins.google on a call's first turn blocked that
    call's event loop for ~1.2 s. Never raises -- build_llm() reports a real
    problem when a call starts."""
    import importlib

    try:
        names = parse_chain(os.environ.get("LLM_CHAIN", DEFAULT_CHAIN))
    except ValueError:
        return
    for name in names:
        try:
            importlib.import_module(_PLUGIN_MODULE[name])
        except Exception:
            logger.warning("could not preload the %s LLM plugin", name, exc_info=True)


def _build_one(name: str):
    # Dispatched by name at call time rather than through a dict of function
    # references, which would freeze the bindings at import.
    if name == "groq":
        return _build_groq()
    if name == "gemini":
        return _build_gemini()
    if name == "sarvam":
        return _build_sarvam()
    raise ValueError(f"no builder for provider {name!r}")


def parse_chain(raw: str) -> list[str]:
    """Parse LLM_CHAIN into provider names, preserving order.

    Raises ValueError naming the offender for an unknown or empty chain --
    a silently dropped fallback is worse than a failed startup.
    """
    names = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not names:
        raise ValueError(
            f"LLM_CHAIN is empty; expected a comma-separated list of "
            f"{sorted(_REQUIRED_KEY)} (default: {DEFAULT_CHAIN!r})"
        )
    unknown = [n for n in names if n not in _REQUIRED_KEY]
    if unknown:
        raise ValueError(
            f"LLM_CHAIN names unknown provider(s) {unknown}; "
            f"supported: {sorted(_REQUIRED_KEY)}"
        )
    seen = set()
    duplicates = [n for n in names if n in seen or seen.add(n)]
    if duplicates:
        raise ValueError(f"LLM_CHAIN repeats provider(s) {duplicates}")
    return names


def build_llm():
    """Build the LLM (or fallback chain) named by LLM_CHAIN."""
    from livekit.agents.llm import FallbackAdapter

    names = parse_chain(os.environ.get("LLM_CHAIN", DEFAULT_CHAIN))

    missing = [
        f"{name} (needs {_REQUIRED_KEY[name]})"
        for name in names
        if not os.environ.get(_REQUIRED_KEY[name], "").strip()
    ]
    if missing:
        raise ValueError(
            "LLM_CHAIN lists provider(s) whose API key is not set: "
            + ", ".join(missing)
        )

    instances = [_build_one(name) for name in names]
    logger.info("llm chain: %s", " -> ".join(names))

    if len(instances) == 1:
        # FallbackAdapter over a single entry buys nothing but indirection.
        return instances[0]

    return FallbackAdapter(
        instances,
        attempt_timeout=float(os.environ.get("LLM_ATTEMPT_TIMEOUT", "4")),
    )
