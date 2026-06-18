# On-Call Copilot — Multi-Agent Incident Response

A 4-agent parallel incident response system. Ingests raw incident signals (logs, alerts, metrics) and returns in under 30 seconds: structured triage, executive summary, comms draft, and post-incident report — all via a single `POST /incident` call.

Adapted from Microsoft's [On-Call Copilot Multi-Agent](https://github.com/leestott/On-Call-Copilot-Multi-Agent) (Lee Stott) to use [HolySheep](https://holysheep.ai) as the LLM relay, deployed to [Fly.io Sprites](https://sprites.dev) for 24/7 persistence.

## Architecture

```
HTTP POST /incident
       ↓
  FastAPI (port 8090)
       ↓
  Orchestrator (asyncio.gather, per-agent timeout 30s)
       ↓
  ┌────────────┬────────────┬────────────┬────────────┐
  │  Triage    │  Summary   │   Comms    │    PIR     │  ← 4 specialists in parallel
  │ deepseek-  │ gemini-    │ claude-    │ deepseek-  │
  │ v4-pro     │ 2.5-flash  │ haiku-4-5  │ v4-pro     │
  └────────────┴────────────┴────────────┴────────────┘
       ↓
  Merged JSON response
```

## Setup (Local Development)

```bash
git clone https://github.com/mnjbold/oncall-copilot.git
cd oncall-copilot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env — set HOLYSHEEP_API_KEY to your HolySheep key
pytest --ignore=tests/test_api.py -q   # 64 tests pass
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Live Deployment

The service runs on Fly.io Sprite `hermes-omega` behind nginx reverse proxy at `https://hermes-omega-bufxd.sprites.app/oncall/`.

```bash
# Health check
curl https://hermes-omega-bufxd.sprites.app/oncall/healthz

# Trigger an incident
curl -X POST https://hermes-omega-bufxd.sprites.app/oncall/incident \
  -H "Content-Type: application/json" \
  -d '{
    "title": "DB connection timeout",
    "service": "checkout-api",
    "alert": "db: connection refused",
    "logs": "2026-06-18 ERROR db timeout",
    "metrics": "cpu=95 memory=80"
  }'
```

Response is a merged JSON dict with all 4 agents' outputs and an `elapsed_ms` field.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `HOLYSHEEP_API_KEY` | Yes | HolySheep relay API key (https://holysheep.ai/app/apikeys) |
| `HOLYSHEEP_BASE_URL` | No | Defaults to `https://api.holysheep.ai/v1` |
| `TRIAGE_MODEL` | No | Defaults to `deepseek-v4-pro` |
| `SUMMARY_MODEL` | No | Defaults to `gemini-2.5-flash` |
| `COMMS_MODEL` | No | Defaults to `claude-haiku-4-5` |
| `PIR_MODEL` | No | Defaults to `deepseek-v4-pro` |
| `PORT` | No | Defaults to `8080` (uvicorn binding port; nginx proxies `/oncall/` to `8090`) |

## API Contract

### `POST /incident`

Request body (Pydantic `IncidentRequest`):
```python
{
    "title": str,       # required
    "service": str,     # required
    "alert": str,
    "logs": str,
    "metrics": str,
}
```

Response (Pydantic `IncidentResponse`):
```python
{
    # Triage
    "suspected_root_causes": [...],
    "immediate_actions": [...],
    "missing_information": [...],
    "runbook_alignment": {...},
    # Summary
    "summary": {...},
    # Comms
    "comms": {...},
    # PIR
    "post_incident_report": {...},
    # Meta
    "elapsed_ms": int,
}
```

Per-agent failures degrade gracefully — the field still exists, but with an `_error` key or empty payload indicating the agent failed.

### `GET /healthz`

Returns `{"status": "ok"}` if the service is up.

## Models

Each specialist agent uses a different model, optimized for cost + capability:

| Agent | Model | Cost (per 1M tokens) | Why |
|---|---|---|---|
| Triage | `deepseek-v4-pro` | $0.29 input / $0.58 output | Strong reasoning, structured JSON output |
| Summary | `gemini-2.5-flash` | $0.04 input / $0.35 output | Fast, long context, narrative generation |
| Comms | `claude-haiku-4-5` | $0.14 input / $0.69 output | Short creative writing, Slack-style tone |
| PIR | `deepseek-v4-pro` | $0.29 input / $0.58 output | Structured timeline + action items |

Per-incident cost on HolySheep: ~$0.001–$0.05 depending on payload size.

## Cost Notes

- 4 parallel LLM calls per incident = ~$0.01–$0.05 total
- Daily cap: HolySheep Starter plan = $20/day hard rate-limit
- Per-agent 30s timeout prevents runaway costs
- Cache writes/reads work on Anthropic models for repeated system prompts

## Maintenance

- **Auto-restart**: uvicorn runs via `nohup` + `disown` on Sprite; Sprite auto-resumes paused VMs on URL hit
- **Logs**: `/tmp/oncall.log` on sprite (visible via `sprite exec -s hermes-omega -- tail /tmp/oncall.log`)
- **Updates**: pull latest, kill old uvicorn, restart: `sprite exec -s hermes-omega -- bash -c 'kill -9 $(pgrep -f uvicorn) && cd /home/sprite/oncall-copilot && ./start.sh &'`

## License

MIT — adapted from Microsoft's reference implementation.

## References

- [Microsoft On-Call Copilot Multi-Agent](https://github.com/leestott/On-Call-Copilot-Multi-Agent)
- [Microsoft Tech Community blog](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/building-a-multi-agent-on-call-copilot-with-microsoft-agent-framework/4499962)
- [HolySheep relay](https://holysheep.ai) — unified access to Claude / GPT / Gemini / DeepSeek / Doubao / MiniMax
- [Fly.io Sprites](https://sprites.dev) — persistent sandbox VMs