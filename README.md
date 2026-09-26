# Voice Agent Platform

A real-time voice agent for field operations: a LiveKit Agents worker
(Sarvam STT/TTS, a configurable LLM fallback chain, Moss + Qdrant context
retrieval) paired with a React/TypeScript console for connecting to a live
voice session, watching its state, transcript, and tool activity in real
time.

The LLM is whatever `LLM_CHAIN` names, tried left to right — `groq,gemini`
by default, with a Sarvam option for India-local inference. See
`agent/app/voice_providers/llm_chain.py`.

This README is the current, verified entry point. `STATUS_REPORT.md` is the
detailed build log from the original backend build session — useful
history, but superseded by this file and by the test results below where
the two differ.

## Layout

```
agent/       FastAPI token service + LiveKit Agents worker (Python)
frontend/    React + TypeScript + Vite console (livekit-client)
docs/adr/    Architecture decision records
render.yaml  Render Blueprint for the backend (see agent/render.yaml)
```

## Verified state

- **Backend**: 146/146 tests passed against a real local PostgreSQL 16
  instance at the time of the original build session. Coverage 83%
  overall; the 5 modules that previously had 0% coverage
  (`agent_entrypoint.py`, `context/embeddings.py`,
  `context/moss_provider.py`, `context/qdrant_provider.py`,
  `tools/definitions.py`) now have dedicated tests.
- **Backend, since then**: the latency-benchmark and LLM-chain work added
  `tests/test_latency_report.py` and `tests/test_llm_chain.py`. 120 tests
  pass without a database (`pytest --ignore=tests/test_auth_identity.py`);
  the DB-backed tests need `DATABASE_URL` and have not been re-run against
  a real Postgres since, so the combined figure above is not current.
  `tests/test_auth_identity.py` additionally fails to import on an
  unrelated pre-existing `ExpiredTokenError` symbol.
- **Frontend**: `npm run lint` and `npm run build` both clean. No automated
  test suite exists yet (no vitest/jest configured).
- **End-to-end locally**: both services started together, real
  `/v1/dev/token` and `/v1/livekit/token` calls made and verified against a
  local Postgres + fake LiveKit credentials (this sandbox has no network
  path to real LiveKit Cloud, Neon, Sarvam, or Moss — see Limitations).
- **CORS**: added and tested this session (`tests/test_cors.py`, 4 tests).
  The token service previously had none, which silently blocks every
  browser request from a different origin than a same-origin CORS-blind
  curl/httpx test could ever catch.

## Local quick start

### 0. Prerequisites

Python 3.12+, Node 18+, Docker, and five accounts (all have a free tier):

| Service | Supplies | Where |
| --- | --- | --- |
| LiveKit Cloud | `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | livekit.io |
| Sarvam AI | `SARVAM_API_KEY` (STT, TTS, and optionally the LLM) | sarvam.ai |
| Groq | `GROQ_API_KEY` | console.groq.com |
| Google AI Studio | `GOOGLE_API_KEY` | aistudio.google.com |
| Moss | `MOSS_PROJECT_ID`, `MOSS_PROJECT_KEY` | your Moss project |

Get the keys first — it is the slow part of setup.

### 1. Postgres and Qdrant

`docker-compose.yml` provisions both, already configured for this project:

```bash
docker-compose up -d
docker-compose ps      # wait for both to report healthy
```

### 2. Configure the backend

```bash
cd agent
cp .env.production.example .env
```

Fill in `agent/.env`. The compose stack above corresponds to:

```bash
DATABASE_URL=postgresql+asyncpg://voiceagent:voiceagent@localhost:5432/voiceagent
QDRANT_URL=http://localhost:6333
JWT_SECRET=any-long-random-string
ENVIRONMENT=development
ALLOWED_ORIGINS=http://localhost:5173
```

plus the five services' keys from the table above.

Two things that have each cost a debugging session:

- The scheme must be **`postgresql+asyncpg://`**, not `postgresql://`.
- **Edit `.env`; do not `export`.** `app/agent_entrypoint.py` calls
  `load_dotenv(override=True)`, so a value in `.env` silently wins over
  anything exported in your shell. Exporting a tuning variable that also
  appears in `.env` looks like it works and does nothing.

### 3. Install and migrate

```bash
python3 -m venv .venv && source .venv/bin/activate
# Windows Git Bash: python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt -r requirements-dev.txt
alembic upgrade head
```

### 4. Preflight (see the section below — do not skip it)

```bash
python preflight_check.py
```

### 5. Frontend

```bash
cd ../frontend
cp .env.example .env     # set VITE_LIVEKIT_URL to your wss:// URL
npm install
```

### 6. Run — three terminals

```bash
# terminal 1 — token service
cd agent && source .venv/bin/activate
uvicorn app.security.token_service:app --reload --port 8000

# terminal 2 — agent worker
cd agent && source .venv/bin/activate
python -m app.agent_entrypoint dev

# terminal 3 — frontend
cd frontend && npm run dev
```

Open http://localhost:5173 and connect; you should hear the greeting.

Run the backend test suite with `pytest tests/` (the DB-backed tests need
`DATABASE_URL` pointed at a real Postgres — see
`agent/.env.production.example` for the full variable reference and format
gotchas).

### Before trusting a live voice session: run the preflight check

This project was built and tested in a sandbox with no network path to
LiveKit Cloud, Neon, Sarvam, or Moss — every piece of code was verified
correct against these services' real SDKs, but never verified *reachable*.
From a machine with normal internet access:

```bash
cd agent
pip install livekit-api asyncpg aiohttp moss
export $(grep -v '^#' .env | xargs)   # load your real credentials
python preflight_check.py
```

It makes one lightweight, authenticated call per service (Postgres, LiveKit
Cloud, Sarvam, Moss) and checks `GOOGLE_API_KEY` is set, and tells you in
~10 seconds which of the five is actually going to work before you spend
time debugging a live session. Nothing is created or left running.

## Measuring latency

The agent can record LiveKit's own per-turn timings, so latency claims are
numbers rather than impressions.

Set `LATENCY_LOG_PATH` in `agent/.env`, have a real conversation, then
aggregate:

```bash
cd agent
python latency_report.py run.jsonl                          # p50/p95/p99 per stage
python latency_report.py before.jsonl after.jsonl           # A/B with deltas
python latency_report.py before.jsonl after.jsonl --markdown
```

Reported stages: end-to-end (user stops speaking → agent starts),
barge-in stop, turn-end decision, LLM first token (which includes the
inline Moss recall, since Moss runs inside `llm_node`), TTS first byte.

Recording is **off unless `LATENCY_LOG_PATH` is set** — nothing is opened
or written otherwise — and `app/latency_log.py` writes timings only, never
transcript text.

On a deployed worker, set `LATENCY_LOG_PATH=stdout`. Rows go to the worker's
logs prefixed `LATENCY_ROW`, and `latency_report.py` reads a log export
directly: it picks out the marked rows and skips every other line.

What makes a comparison real rather than noise:

- **Say the same things in both runs, in the same order.** Latency tracks
  reply length and question difficulty; different scripts compare nothing.
- **At least ~20 turns.** Below that the tool warns you that p95 is the
  slowest turn rather than a tail, and it is right — quote p50.
- **Interrupt the agent deliberately, 5–6 times.** Only real interruptions
  populate `barge_in_latency`.

Example: comparing LLM providers by editing only `LLM_CHAIN` between runs.

```bash
LATENCY_LOG_PATH=groq.jsonl    LLM_CHAIN=groq,gemini
LATENCY_LOG_PATH=sarvam.jsonl  LLM_CHAIN=sarvam,groq,gemini
```

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `KeyError: 'GROQ_API_KEY'` at startup | key missing from `.env`, or drop that provider from `LLM_CHAIN` |
| `LLM_CHAIN lists provider(s) whose API key is not set` | working as intended — fix the key or the chain, rather than losing a fallback silently |
| `LLM_CHAIN names unknown provider(s)` | supported names are `groq`, `gemini`, `sarvam` |
| A tuning variable has no effect | it is also set in `.env`, which overrides your shell (see step 2) |
| Browser connects but there is no audio | `VITE_LIVEKIT_URL` wrong, or the agent worker (terminal 2) is not running |
| CORS error in the browser console | `ALLOWED_ORIGINS` must match the frontend origin exactly, e.g. `http://localhost:5173` |
| Agent never responds | check terminal 2 — rate limits, auth failures, and provider fallbacks surface there |
| `ModuleNotFoundError: No module named 'alembic.config'` | `PYTHONPATH=.` shadows the installed package — don't set it. Run `alembic upgrade head` directly; `alembic.ini` already contains `prepend_sys_path = .` which lets Alembic find `app.*` on its own |

## Deploying: Vercel (frontend) + Render (backend)

- `agent/render.yaml` — Blueprint defining the token service (web) and
  agent worker (background worker) as two Render services. Every secret is
  `sync: false`; fill real values in Render's dashboard, not in the file.
- Frontend: Vercel auto-detects Vite. Set **Root Directory** to `frontend`
  (the one non-default setting for this monorepo), and set
  `VITE_API_BASE_URL` / `VITE_LIVEKIT_URL` as environment variables.
- Full walkthrough, including two real bugs found and fixed while checking
  an actual `.env` against this code (a bad `DATABASE_URL` format for
  asyncpg, and a missing `GOOGLE_API_KEY`), is in
  `docs/voice-agent-platform-local-start-and-deployment.pdf`.

## Known limitations

- **Production authentication is not implemented.** `/v1/dev/token` mints a
  token for any `tenant_slug`/`external_id` with no login check — fine
  for development, a real gap the moment a deployed URL is reachable.
  `ENVIRONMENT=development` must stay off in any environment you don't
  fully control access to.
- **`QDRANT_LOCATION=:memory:`** loses all upserted knowledge on every
  worker restart and doesn't share state across multiple worker instances.
  Swap for a persistent Qdrant (Cloud or self-hosted) for anything beyond a
  single always-on demo worker.
- **No live verification against real external services** (LiveKit Cloud,
  Neon, Sarvam, Moss) has been possible from this development sandbox — its
  network is allowlisted to package registries only, confirmed via
  `x-deny-reason: host_not_allowed` on every attempt. Everything above the
  "Verified state" line was checked locally or by reading the actual SDK
  source. Run `agent/preflight_check.py` from a machine with real network
  access to get an actual answer for each service before assuming a live
  voice session will work.
- `context/orchestrator.py` is at 31% test coverage — the concurrent
  moss/qdrant/live-api retrieval path is still largely untested.
- `context/live_api.py` defines a `LiveOperationalAPI` class nothing
  imports — dead code (an identical class lives inline in
  `orchestrator.py`, which is what's actually used). Worth deleting.
