#!/usr/bin/env python3
"""
Preflight check for voice-agent-platform's real external services.

WHY THIS EXISTS: this project has been developed and tested in a sandbox
with no network path to LiveKit Cloud, Neon, Sarvam, or Moss -- outbound
traffic there is allowlisted to package registries only. Every attempt to
reach those hosts from that sandbox is intercepted by the sandbox's own
egress proxy and returns HTTP 403 with header "x-deny-reason:
host_not_allowed" and body "Host not in allowlist: <host>..." -- confirmed
directly against each host. That is an ENVIRONMENT / NETWORK BLOCK, never a
credential failure: the real service is never actually reached, so this
script explicitly detects that exact signature and reports NETWORK BLOCKED
rather than misreading a proxy's 403 as the real service rejecting the key.

This script exists so you can get a real, authenticated answer for each
service from a machine that isn't sandboxed this way.

WHAT IT DOES: loads a real .env, then makes ONE lightweight, real,
authenticated request per service using the project's own actual
endpoints/SDKs (not guessed ones) -- the same LiveKit REST call the token
service depends on, the same Sarvam TTS endpoint app/voice_providers/sarvam
uses, the same DATABASE_URL the app's SQLAlchemy engine uses, the same Moss
SDK app/context/moss_provider.py uses. No LiveKit room is left running, no
data is written anywhere, nothing is deleted.

For the LLM it sends a 1-token completion to every provider in LLM_CHAIN,
using the model the worker will actually call (llm_chain.model_for). A
retired model or a rejected key is otherwise invisible until a live call:
FallbackAdapter quietly routes around the broken provider. Each passing
check is timed (cold, then warm on the same connection), and
LLM_PROBE_MODELS="groq:model-a,groq:model-b" times candidate models too --
the LLM's first token is the largest stage of a voice reply, so pick the
model from these numbers.

On the deployed worker it runs at every boot, ahead of the agent, so these
results are the first thing in the worker's logs.

WHAT IT NEVER DOES: print an API key, secret, JWT, or database password.
Every credential is masked to its first 4 and last 2 characters.

Usage (Windows, macOS, Linux -- all the same):
    cd agent
    pip install livekit-api asyncpg aiohttp moss
    python preflight_check.py                 # loads .env from this directory
    python preflight_check.py "path/to/.env"  # or point at one explicitly
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import statistics
import sys
import time
import urllib.parse
from typing import NamedTuple

from app.voice_providers import llm_chain

# Read by the agent worker. The LLM keys are added per LLM_CHAIN at runtime.
WORKER_VARS = [
    "DATABASE_URL", "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET",
    "MOSS_PROJECT_ID", "MOSS_PROJECT_KEY", "SARVAM_API_KEY",
]
# Read only by the token service -- MISSING is expected on a worker-only host.
TOKEN_SERVICE_VARS = ["JWT_SECRET", "JWT_ISSUER", "ENVIRONMENT"]


class CheckResult(NamedTuple):
    status: str  # "AUTHENTICATED" | "REJECTED" | "NETWORK BLOCKED" | "MISSING" | "ERROR"
    detail: str  # human-readable, never contains a secret
    ms: float | None = None  # LLM checks: warm reply time, when the check passed


def mask(value: str) -> str:
    if len(value) <= 8:
        return "…"
    return f"{value[:4]}…{value[-2:]} (len={len(value)})"


def load_env_file(path: str) -> dict[str, str]:
    """Minimal .env parser -- no external dependency. Does not overwrite
    variables already present in the real environment."""
    loaded: dict[str, str] = {}
    if not os.path.exists(path):
        return loaded
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
            loaded[key] = value
    return loaded


async def egress_probe(host: str) -> CheckResult | None:
    """Real HTTP-level probe for the exact egress-proxy signature this
    sandbox (and possibly a corporate firewall doing the same thing) uses:
    HTTP 403 with header x-deny-reason: host_not_allowed. A DNS failure or
    a raw connection failure is also a network block. Anything else (a
    normal-looking response of any status) means the request actually
    reached something -- callers proceed to their real, service-specific
    request in that case. Returns None when nothing indicates a block."""
    try:
        import aiohttp
    except ImportError:
        return None  # let the caller's own import-check report this

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://{host}/", timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.headers.get("x-deny-reason") == "host_not_allowed":
                    return CheckResult("NETWORK BLOCKED", f"egress proxy blocked {host} (host_not_allowed)")
                body_start = (await resp.text())[:120] if resp.status == 403 else ""
                if "not in allowlist" in body_start.lower():
                    return CheckResult("NETWORK BLOCKED", f"egress proxy blocked {host}: {body_start}")
        return None
    except socket.gaierror as e:
        return CheckResult("NETWORK BLOCKED", f"DNS resolution failed for {host}: {e}")
    except (TimeoutError, ConnectionRefusedError, OSError) as e:
        return CheckResult("NETWORK BLOCKED", f"cannot reach {host}: {e}")
    except Exception:
        return None  # anything else -- let the real request attempt and speak for itself


async def check_database() -> CheckResult:
    url = os.environ.get("DATABASE_URL")
    if not url:
        return CheckResult("MISSING", "DATABASE_URL not set")

    if "sslmode" in url or "channel_binding" in url:
        return CheckResult(
            "ERROR",
            "DATABASE_URL has sslmode=/channel_binding= query params -- asyncpg (this app's driver) "
            "rejects those outright with TypeError. Required format: "
            "postgresql+asyncpg://USER:PASS@HOST/DB?ssl=require -- fixing the URL is required before "
            "this check can mean anything; not re-attempted here since it would only reproduce that error.",
        )

    parsed = urllib.parse.urlsplit(url.replace("postgresql+asyncpg://", "postgresql://", 1))
    try:
        socket.setdefaulttimeout(6)
        with socket.create_connection((parsed.hostname, parsed.port or 5432), timeout=6):
            pass
    except socket.gaierror as e:
        return CheckResult("NETWORK BLOCKED", f"DNS resolution failed for {parsed.hostname}: {e}")
    except (TimeoutError, ConnectionRefusedError, OSError) as e:
        return CheckResult("NETWORK BLOCKED", f"TCP connect to {parsed.hostname}:{parsed.port or 5432} failed: {e}")

    try:
        import asyncpg
    except ImportError:
        return CheckResult("ERROR", "asyncpg not installed (pip install asyncpg)")

    dsn = url.replace("postgresql+asyncpg://", "postgresql://", 1)
    try:
        conn = await asyncio.wait_for(asyncpg.connect(dsn=dsn), timeout=10)
        result = await conn.fetchval("SELECT 1")
        await conn.close()
        return CheckResult("AUTHENTICATED", f"SELECT 1 -> {result}")
    except asyncio.TimeoutError:
        return CheckResult("NETWORK BLOCKED", "connection timed out mid-handshake")
    except Exception as e:
        msg = str(e).lower()
        if "password" in msg or "authentication" in msg:
            return CheckResult("REJECTED", f"{type(e).__name__}: {e}")
        return CheckResult("ERROR", f"{type(e).__name__}: {e}")


async def check_livekit() -> CheckResult:
    url = os.environ.get("LIVEKIT_URL")
    key = os.environ.get("LIVEKIT_API_KEY")
    secret = os.environ.get("LIVEKIT_API_SECRET")
    if not all([url, key, secret]):
        return CheckResult("MISSING", "LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET not fully set")

    host = urllib.parse.urlsplit(url.replace("wss://", "https://").replace("ws://", "http://")).hostname
    blocked = await egress_probe(host)
    if blocked:
        return blocked

    try:
        from livekit import api
    except ImportError:
        return CheckResult("ERROR", "livekit-api not installed (pip install livekit-api)")

    lk = api.LiveKitAPI(url=url, api_key=key, api_secret=secret)
    try:
        rooms = await asyncio.wait_for(lk.room.list_rooms(api.ListRoomsRequest()), timeout=10)
        return CheckResult("AUTHENTICATED", f"ListRooms succeeded -- {len(rooms.rooms)} room(s) currently active")
    except Exception as e:
        msg = str(e)
        if "403" in msg or "401" in msg or "unauthorized" in msg.lower() or "invalid" in msg.lower():
            return CheckResult("REJECTED", f"{type(e).__name__}: {e}")
        return CheckResult("ERROR", f"{type(e).__name__}: {e}")
    finally:
        await lk.aclose()


def sarvam_tts_payload() -> dict:
    """The REST request livekit-plugins-sarvam sends, for the voice the agent
    speaks with (factory.tts_voice -- the same function build_tts uses)."""
    from app.voice_providers.factory import tts_voice

    model, speaker, language = tts_voice()
    return {"text": "preflight check", "target_language_code": language, "speaker": speaker, "model": model}


async def check_sarvam_tts() -> CheckResult:
    """Same endpoint and request shape as the Sarvam plugin, with the agent's
    own model, speaker and language -- a retired voice fails here, not mid-call."""
    key = os.environ.get("SARVAM_API_KEY")
    if not key:
        return CheckResult("MISSING", "SARVAM_API_KEY not set")

    blocked = await egress_probe("api.sarvam.ai")
    if blocked:
        return blocked

    try:
        import aiohttp
    except ImportError:
        return CheckResult("ERROR", "aiohttp not installed (pip install aiohttp)")

    payload = sarvam_tts_payload()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.sarvam.ai/text-to-speech",
                headers={"api-subscription-key": key, "Content-Type": "application/json"},
                json=payload,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.headers.get("x-deny-reason") == "host_not_allowed":
                    return CheckResult("NETWORK BLOCKED", "egress proxy blocked api.sarvam.ai (host_not_allowed)")
                if resp.status == 200:
                    return CheckResult(
                        "AUTHENTICATED",
                        f"POST /text-to-speech -> 200 OK ({payload['model']}, speaker {payload['speaker']})",
                    )
                if resp.status in (401, 403):
                    return CheckResult("REJECTED", f"HTTP {resp.status} -- key rejected by Sarvam")
                body = (await resp.text())[:150]
                return CheckResult("ERROR", f"HTTP {resp.status}: {body}")
    except Exception as e:
        return CheckResult("ERROR", f"{type(e).__name__}: {e}")


async def check_moss() -> CheckResult:
    project_id = os.environ.get("MOSS_PROJECT_ID")
    project_key = os.environ.get("MOSS_PROJECT_KEY")
    if not all([project_id, project_key]):
        return CheckResult("MISSING", "MOSS_PROJECT_ID / MOSS_PROJECT_KEY not fully set")

    # Moss's actual API hosts (found by inspecting the compiled moss_core
    # extension -- not documented in the Python package itself).
    for host in ("service.usemoss.dev", "models.moss.link"):
        blocked = await egress_probe(host)
        if blocked:
            return blocked

    try:
        from moss import MossClient
    except ImportError:
        return CheckResult("ERROR", "moss not installed (pip install moss)")

    try:
        client = MossClient(project_id, project_key)
        indexes = await asyncio.wait_for(client.list_indexes(), timeout=10)
        count = len(indexes) if hasattr(indexes, "__len__") else "?"
        return CheckResult("AUTHENTICATED", f"list_indexes() succeeded -- {count} index(es)")
    except Exception as e:
        msg = str(e).lower()
        if "401" in msg or "403" in msg or "unauthorized" in msg or "invalid" in msg:
            return CheckResult("REJECTED", f"{type(e).__name__}: {e}")
        return CheckResult("ERROR", f"{type(e).__name__}: {e}")


def llm_request(provider: str, model: str, key: str) -> tuple[str, str, dict, dict]:
    """(host, url, headers, json) for a 1-token completion, matching how the
    agent's own clients call each provider. Keys go in headers, never the URL,
    so an exception message can't carry one into the logs."""
    ping = [{"role": "user", "content": "ping"}]
    if provider == "groq":
        body = {"model": model, "messages": ping, "max_completion_tokens": 16}
        effort = llm_chain.groq_reasoning_effort(model)
        if effort:
            # Sent exactly as the worker sends it: Groq rejects the parameter
            # for models that don't reason, and that should fail here.
            body["reasoning_effort"] = effort
        return (
            "api.groq.com",
            "https://api.groq.com/openai/v1/chat/completions",
            {"Authorization": f"Bearer {key}"},
            body,
        )
    if provider == "gemini":
        return (
            "generativelanguage.googleapis.com",
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            {"x-goog-api-key": key},
            {"contents": [{"parts": [{"text": "ping"}]}], "generationConfig": {"maxOutputTokens": 16}},
        )
    if provider == "sarvam":
        # livekit-plugins-sarvam serves sarvam-105b-conversations from /v1 and
        # every other model from /v2 (llm/client.py::_resolve_base_url).
        version = "v1" if model == "sarvam-105b-conversations" else "v2"
        return (
            "api.sarvam.ai",
            f"https://api.sarvam.ai/{version}/chat/completions",
            {"api-subscription-key": key},
            {"model": model, "messages": ping, "max_tokens": 16},
        )
    raise ValueError(f"no preflight request for LLM provider {provider!r}")


_BAD_KEY_MARKERS = ("invalid_api_key", "invalid api key", "api key not valid", "api_key_invalid")
_BAD_MODEL_MARKERS = (
    "model_not_found", "model_decommissioned", "decommissioned", "does not exist",
    "is not found", "not supported for generatecontent", "invalid model", "unsupported model",
)


def classify_llm_response(provider: str, model: str, status: int, body: str) -> CheckResult:
    """Turn a provider's reply into a verdict that names the fix: a rejected
    key and an unavailable model look alike from the agent (the provider just
    fails) but need different changes."""
    lowered = body.lower()
    if status == 200:
        return CheckResult("AUTHENTICATED", f"{model} answered a 1-token completion")
    if status in (401, 403) or any(m in lowered for m in _BAD_KEY_MARKERS):
        return CheckResult(
            "REJECTED", f"HTTP {status} -- {provider} rejected {llm_chain.key_env(provider)}"
        )
    if status == 404 or any(m in lowered for m in _BAD_MODEL_MARKERS):
        return CheckResult(
            "REJECTED",
            f"HTTP {status} -- model {model!r} is not available on {provider}; set "
            f"{llm_chain.model_env(provider)} to a current model. {body[:150]}",
        )
    if status == 429:
        return CheckResult(
            "ERROR",
            f"HTTP 429 -- the key works but {provider} is rate-limiting it right now; "
            f"FallbackAdapter will route around {provider} until it recovers",
        )
    return CheckResult("ERROR", f"HTTP {status}: {body[:150]}")


async def _get_json(url: str, headers: dict) -> tuple[int, str]:
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            return resp.status, await resp.text()


async def available_models(provider: str, key: str, *, get=None) -> list[str] | None:
    """Chat-capable model IDs this key can use, or None if the provider has no
    listing endpoint or the listing fails. Turns "model X is gone" into
    "use one of these"."""
    import json

    get = get or _get_json
    try:
        if provider == "groq":
            status, body = await get("https://api.groq.com/openai/v1/models", {"Authorization": f"Bearer {key}"})
            if status != 200:
                return None
            skip = ("whisper", "tts", "guard", "orpheus", "distil")
            return sorted(m["id"] for m in json.loads(body).get("data", []) if not any(s in m["id"] for s in skip))
        if provider == "gemini":
            status, body = await get(
                "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200", {"x-goog-api-key": key}
            )
            if status != 200:
                return None
            return sorted(
                m["name"].removeprefix("models/") for m in json.loads(body).get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
            )
    except Exception:
        return None
    return None


@contextlib.asynccontextmanager
async def _http_post():
    """A `post(url, headers, payload)` over ONE pooled connection, the way the
    agent's own LLM client holds it for a whole call -- so the timed requests
    after the first measure the provider, not a fresh TLS handshake."""
    import aiohttp

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
        async def post(url: str, headers: dict, payload: dict) -> tuple[int, str]:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.headers.get("x-deny-reason") == "host_not_allowed":
                    return -1, "host_not_allowed"
                return resp.status, await resp.text()

        yield post


# Requests per passing LLM check: the first (cold: DNS + TLS) decides the
# verdict, the rest reuse its connection and give the warm time a live call sees.
TIMING_SAMPLES = 3


async def _timed(post, url: str, headers: dict, payload: dict) -> tuple[int, str, float]:
    start = time.perf_counter()
    status, body = await post(url, headers, payload)
    return status, body, (time.perf_counter() - start) * 1000


async def check_llm(provider: str, *, model: str | None = None, post=None, get=None) -> CheckResult:
    """A tiny completion against `provider` with the model the worker will
    call (or `model`, for a latency probe), timed. `post`/`get` are
    injectable for tests; they default to aiohttp."""
    key = os.environ.get(llm_chain.key_env(provider), "").strip()
    if not key:
        return CheckResult("MISSING", f"{llm_chain.key_env(provider)} not set")

    model = model or llm_chain.model_for(provider)
    host, url, headers, payload = llm_request(provider, model, key)

    # Listing follows the transport: real network in real use, stubbed with post in tests.
    can_list = post is None or get is not None
    if post is not None:
        return await _run_llm_check(provider, model, key, url, headers, payload, post, get, can_list)

    blocked = await egress_probe(host)
    if blocked:
        return blocked
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        return CheckResult("ERROR", "aiohttp not installed (pip install aiohttp)")
    async with _http_post() as post:
        return await _run_llm_check(provider, model, key, url, headers, payload, post, get, can_list)


async def _run_llm_check(provider, model, key, url, headers, payload, post, get, can_list) -> CheckResult:
    try:
        status, body, cold_ms = await _timed(post, url, headers, payload)
    except Exception as e:
        return CheckResult("ERROR", f"{type(e).__name__}: {e}")
    if status == -1:
        host = urllib.parse.urlparse(url).hostname
        return CheckResult("NETWORK BLOCKED", f"egress proxy blocked {host} (host_not_allowed)")
    result = classify_llm_response(provider, model, status, body)
    if result.status == "AUTHENTICATED":
        warm = []
        for _ in range(TIMING_SAMPLES - 1):
            try:
                status, _, ms = await _timed(post, url, headers, payload)
            except Exception:
                break
            if status != 200:
                break
            warm.append(ms)
        ms = statistics.median(warm) if warm else cold_ms
        return result._replace(
            detail=f"{model} replied in {ms:.0f} ms warm ({cold_ms:.0f} ms cold)", ms=ms
        )
    if can_list and result.status == "REJECTED" and llm_chain.model_env(provider) in result.detail:
        models = await available_models(provider, key, get=get)
        if models:
            result = result._replace(detail=f"{result.detail} Available: {', '.join(models[:30])}")
    return result


def llm_checks() -> list[tuple[str, object]]:
    """(label, coroutine-or-result) for each provider in LLM_CHAIN, in order."""
    try:
        providers = llm_chain.parse_chain(os.environ.get("LLM_CHAIN", llm_chain.DEFAULT_CHAIN))
    except ValueError as e:
        return [("LLM chain", CheckResult("ERROR", str(e)))]
    return [(f"LLM {p} ({llm_chain.model_for(p)})", check_llm(p)) for p in providers]


def probe_checks() -> list[tuple[str, object]]:
    """(label, coroutine-or-result) for each "provider:model" in LLM_PROBE_MODELS.

    Times candidate models from where the worker actually runs, so choosing
    GROQ_MODEL is a measurement rather than a guess. Informational: a probe
    never changes the PASS/FAIL verdict."""
    raw = os.environ.get("LLM_PROBE_MODELS", "")
    out: list[tuple[str, object]] = []
    for entry in (e.strip() for e in raw.split(",")):
        if not entry:
            continue
        provider, sep, model = entry.partition(":")
        provider, model = provider.strip().lower(), model.strip()
        if not sep or not model or provider not in ("groq", "gemini", "sarvam"):
            out.append((f"probe {entry}", CheckResult("ERROR", "expected provider:model, e.g. groq:llama-3.1-8b-instant")))
            continue
        out.append((f"probe {provider} ({model})", check_llm(provider, model=model)))
    return out


async def main() -> int:
    env_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), ".env")
    loaded = load_env_file(env_path)

    print("=" * 60)
    print("ENVIRONMENT VARIABLE CHECK  (values masked)")
    print("=" * 60)
    chain = os.environ.get("LLM_CHAIN", llm_chain.DEFAULT_CHAIN)
    print(f"{'LLM_CHAIN':20} {chain}")
    try:
        chain_keys = [llm_chain.key_env(p) for p in llm_chain.parse_chain(chain)]
    except ValueError:
        chain_keys = []
    for name in WORKER_VARS + chain_keys:
        val = os.environ.get(name)
        print(f"{name:20} {'PRESENT ' + mask(val) if val else 'MISSING'}")
    print("-- token service only (MISSING is expected on a worker-only host) --")
    for name in TOKEN_SERVICE_VARS:
        val = os.environ.get(name)
        print(f"{name:20} {'PRESENT ' + mask(val) if val else 'MISSING'}")
    if not loaded:
        print(f"\n(no .env file found at {env_path} -- using process environment only)")

    print()
    print("=" * 60)
    print("LIVE SERVICE CHECKS  (real authenticated requests)")
    print("=" * 60)

    llm = llm_checks()
    checks = [
        ("Database (DATABASE_URL)", check_database()),
        ("LiveKit", check_livekit()),
        ("Sarvam TTS", check_sarvam_tts()),
        ("Moss", check_moss()),
        *llm,
    ]
    results: dict[str, CheckResult] = {}
    for name, pending in checks:
        result = pending if isinstance(pending, CheckResult) else await pending
        results[name] = result
        print(f"[{result.status:16}] {name}: {result.detail}")

    probes = probe_checks()
    if probes:
        print()
        print("LLM LATENCY PROBE  (LLM_PROBE_MODELS -- informational, never affects PASS/FAIL)")
        for name, pending in probes:
            result = pending if isinstance(pending, CheckResult) else await pending
            print(f"[{result.status:16}] {name}: {result.detail}")

    print()
    print("REAL VOICE PIPELINE STATUS")
    print("=" * 27)

    def line(name: str, key: str) -> None:
        r = results[key]
        verdict = "PASS" if r.status == "AUTHENTICATED" else "FAIL"
        print(f"{name}: {verdict}  ({r.status})")

    line("Database", "Database (DATABASE_URL)")
    line("LiveKit", "LiveKit")
    line("Sarvam TTS", "Sarvam TTS")
    line("Moss", "Moss")
    for name, _ in llm:
        line(name, name)

    network_blocked = any(r.status == "NETWORK BLOCKED" for r in results.values())
    rejected = any(r.status == "REJECTED" for r in results.values())
    all_pass = all(r.status == "AUTHENTICATED" for r in results.values())

    print()
    print("LOCAL TESTS: PASS  (env var check + classification logic ran without error)")
    if all_pass:
        print("REAL EXTERNAL TESTS: PASS")
    elif network_blocked and not rejected:
        print("REAL EXTERNAL TESTS: BLOCKED  (one or more services unreachable from this network)")
    else:
        print("REAL EXTERNAL TESTS: FAIL")
    print("END-TO-END VOICE: NOT VERIFIED  (this script checks each service independently, not a live")
    print("                   browser -> mic -> LiveKit -> agent -> STT -> LLM -> TTS -> browser round trip)")
    print()
    print("FINAL:")
    if all_pass:
        print("VOICE PIPELINE WORKING: PARTIALLY VERIFIED (every service authenticated; run an actual")
        print("                         session end-to-end to confirm the full round trip)")
    elif network_blocked and not rejected:
        print("VOICE PIPELINE WORKING: NOT VERIFIED (network-blocked -- rerun on a network that can reach")
        print("                         these hosts to get a real answer)")
    else:
        print("VOICE PIPELINE WORKING: NO")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
