# On-Call Copilot

Multi-agent incident-response copilot that turns raw alert/incident context into
parallel specialist outputs (Triage, Summary, Comms, PIR) via the HolySheep LLM
relay, exposed behind a FastAPI HTTP API.

## Architecture

```
                ┌─────────────────────────┐
   HTTP POST    │   FastAPI (port 8080)   │
  /incident ──▶ │  app/main.py            │
                └─────────┬───────────────┘
                          │
                          ▼
                ┌─────────────────────────┐
                │  Orchestrator           │
                │  app/orchestrator.py    │
                │  asyncio.gather(...)    │
                └────┬─────┬─────┬────┬───┘
                     ▼     ▼     ▼    ▼
                 Triage Summary Comms PIR
                  (agents/*.py — specialist prompts)
                     │     │     │    │
                     └─────┴─────┴────┘
                           │
                           ▼
                ┌─────────────────────────┐
                │  HolySheep relay        │
                │  OpenAI-compatible      │
                │  POST /v1/chat/         │
                │      completions        │
                └─────────────────────────┘
```

Four specialist agents run concurrently:

- **Triage** — severity classification (SEV1–SEV4), blast radius, immediate actions.
- **Summary** — plain-English executive summary of the incident.
- **Comms** — stakeholder comms (status page, exec update, customer note).
- **PIR** — post-incident-review skeleton (timeline, root-cause hypotheses, action items).

## Project layout

```
oncall-copilot/
├── app/
│   ├── __init__.py
│   └── main.py              # FastAPI app (port 8080)
├── agents/
│   ├── __init__.py
│   ├── base.py              # shared LLM-call helper (httpx + HolySheep)
│   ├── triage.py            # Triage agent
│   ├── summary.py           # Summary agent
│   ├── comms.py             # Comms agent
│   └── pir.py               # PIR agent
├── tests/
│   ├── __init__.py
│   ├── test_agents.py
│   └── test_api.py
├── .venv/                   # Python virtualenv (not committed)
├── .env.example             # template for required env vars
├── .gitignore
├── requirements.txt
└── CLAUDE.md
```

## Runtime

- **Host**: Fly.io Sprite `hermes-omega` (Ubuntu 25.10, x86_64)
- **URL**: https://hermes-omega-bufxd.sprites.app *(currently public — lock to private after deploy)*
- **Python**: 3.13.7
- **Framework**: FastAPI + Uvicorn
- **Port**: 8080
- **LLM relay**: https://api.holysheep.ai/v1 (OpenAI-compatible `/chat/completions`)

## Commands

All commands assume the sprite working directory `/home/sprite/oncall-copilot`.

### Sprite exec (remote)

```bash
# Open a shell on the sprite
sprite exec -s hermes-omega -- bash

# Run a one-off command on the sprite
sprite exec -s hermes-omega -- bash -c \"<cmd>\"
```

### Python venv

```bash
source .venv/bin/activate        # activate venv
pip install -r requirements.txt  # install deps
deactivate                       # exit venv
```

### Run the API

```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

### Tests

```bash
source .venv/bin/activate
pytest -q                        # full suite
pytest tests/test_agents.py -q   # single file
```

## Environment variables

Defined in `.env` (never committed; see `.env.example` for the template):

| Var                  | Purpose                                            |
|----------------------|----------------------------------------------------|
| `HOLYSHEEP_API_KEY`  | Bearer token for the HolySheep relay               |
| `HOLYSHEEP_BASE_URL` | Base URL (default `https://api.holysheep.ai/v1`)   |
| `TRIAGE_MODEL`       | Model id for the Triage agent                      |
| `SUMMARY_MODEL`      | Model id for the Summary agent                     |
| `COMMS_MODEL`        | Model id for the Comms agent                       |
| `PIR_MODEL`          | Model id for the PIR agent                         |
| `PORT`               | HTTP port (default `8080`)                         |

Load with a `.env` reader inside `app/main.py` (e.g. `python-dotenv` if added, or
read via `os.environ` after sourcing the file in the shell).

## API contract

`POST /incident`

Request:
```json
{
  \"title\": \"Checkout API 5xx spike\",
  \"service\": \"checkout-api\",
  \"severity_hint\": \"SEV2\",
  \"context\": \"Error rate climbed from 0.4% to 12% at 14:02 UTC after deploy v2.41.\",
  \"logs\": \"... optional log excerpt ...\",
  \"metrics\": \"... optional metric snapshot ...\"
}
```

Response (200):
```json
{
  \"triage\": { \"severity\": \"SEV2\", \"blast_radius\": \"...\", \"immediate_actions\": [...] },
  \"summary\": \"Plain-English incident summary...\",
  \"comms\":   { \"status_page\": \"...\", \"exec_update\": \"...\", \"customer_note\": \"...\" },
  \"pir\":     { \"timeline\": [...], \"hypotheses\": [...], \"action_items\": [...] }
}
```

`GET /healthz` → `{\"status\": \"ok\"}` (for sprite health checks).

## Conventions

- **Parallelism** — all four agents MUST be dispatched concurrently with
  `asyncio.gather(*tasks)`. Never call them serially.
- **LLM calls** — go through one helper in `agents/base.py`; do not duplicate
  httpx client construction per agent.
- **Timeouts** — each agent call should be wrapped with a sensible timeout
  (e.g. 30s) and degrade gracefully (return a partial / fallback dict).
- **Pydantic** — every request/response model lives in `app/main.py` (or a
  sibling `app/models.py`) and is reused by tests.
- **No secrets in code** — read everything from environment variables.
- **Tests** — pytest + pytest-asyncio. Mock the LLM call (httpx `AsyncClient`
  `transport`) rather than hitting HolySheep in CI.

## Git / GitHub

- Repo: https://github.com/mnjbold/oncall-copilot
- Branch: `main`
- Default branch protection will be added after first deploy.
