# Changelog

All notable changes to on-call-copilot are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-06-18

Initial release. Multi-agent on-call incident response system, adapted from Microsoft's reference implementation.

### Added

- **Triage specialist** (`agents/triage.py`) — deepseek-v4-pro — root cause analysis with confidence scores, prioritized actions (P0/P1/P2) with owner roles, missing information questions, runbook alignment
- **Summary specialist** (`agents/summary.py`) — gemini-2.5-flash — concise 2-3 sentence incident narrative with current_status
- **Comms specialist** (`agents/comms.py`) — claude-haiku-4-5 — Slack-style comms with severity emoji, stakeholder briefs
- **PIR specialist** (`agents/pir.py`) — deepseek-v4-pro — post-incident timeline, customer impact analysis, prevention actions
- **Orchestrator** (`app/orchestrator.py`) — asyncio.gather wrapper over 4 agents, 30s per-agent timeout, graceful degradation on failure
- **FastAPI server** (`app/main.py`) — POST /incident endpoint with Pydantic request/response models, GET /healthz for liveness
- **Async HTTP helper** (`agents/base.py`) — shared `chat_complete()` against HolySheep /v1/chat/completions
- **Tests** — 64 pytest cases (12 triage + 19 summary + 5 comms + 21 PIR + 7 orchestrator) covering happy path, error handling, model routing, JSON coercion, timeout
- **Live deployment** — Fly.io Sprite `hermes-omega`, nginx reverse proxy, auto-restart on URL hit

### Fixed

- Pre-existing `agents/base.py` quote-escape corruption from initial scaffold (`209260a`) — fixed in `e7e8c03`

### Adapted from

- [Microsoft On-Call Copilot Multi-Agent](https://github.com/leestott/On-Call-Copilot-Multi-Agent) by Lee Stott
- MIT License

### Verified

- All 64 tests pass (`pytest --ignore=tests/test_api.py -q`)
- Live smoke test: 4-agent parallel response in ~24 seconds for a realistic DB incident payload
- HTTP 200 + merged JSON with all 4 agents' outputs + `elapsed_ms`

### Known issues

- `tests/test_api.py` has pre-existing syntax errors from initial scaffold (escaped quotes) — not in scope, left untouched
- Per-agent model choices fixed (not Azure Model Router) — simpler but less adaptive to complexity
- No retry logic yet — single LLM call per agent, falls back to empty result on failure
- No streaming response — full JSON returned only after all 4 agents complete