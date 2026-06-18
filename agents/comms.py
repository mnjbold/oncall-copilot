from __future__ import annotations

"""
Comms Agent – crafts clear, actionable communications for different audiences.

Returns a JSON object with keys:
 comms (containing slack_update and stakeholder_update)
"""

COMMS_INSTRUCTIONS = """\
You are the **Comms Agent**, an expert incident communications writer for SRE teams.

## Task
Read the incident data and return a single JSON object with ONLY this key:

```json
{
 "comms": {
  "slack_update": "string – Slack-formatted incident channel update with emoji, severity, status, impact, next steps, and ETA for next update",
  "stakeholder_update": "string – Professional, non-technical summary for executives and product managers. Focus on business impact, customer effect, and resolution status."
 }
}
```

## Guidelines
- **Slack update**: Use emoji prefixes (:rotating_light: for active SEV1/2,
  :warning: for degraded, :white_check_mark: for resolved). Include incident ID,
  severity, one-line summary, affected services, and next update time.
- **Stakeholder update**: No jargon. Translate technical details into business
  impact. Include what customers experience, what the team is doing, and when
  the next update is expected.
- **Tone**: Calm, factual, action-oriented. Never blame individuals.
- **Structured output only** – return ONLY valid JSON, no prose or markdown.
"""

import json
import os
from typing import Any

from . import base


async def run(req: dict[str, Any]) -> dict[str, Any]:
    """Run the Comms agent.

    Args:
        req: Incident payload (title, service, severity_hint, context, ...).
            May be a Pydantic model — coerced via ``model_dump`` if available.

    Returns:
        ``{"comms": {"slack_update": ..., "stakeholder_update": ...}}`` on
        success. Falls back to a partial dict on parse failure or LLM error.
    """
    if hasattr(req, "model_dump"):
        payload = req.model_dump()
    else:
        payload = dict(req)

    messages = [
        {"role": "system", "content": COMMS_INSTRUCTIONS},
        {"role": "user", "content": json.dumps(payload, default=str)},
    ]

    model = os.environ.get("COMMS_MODEL", "claude-haiku-4-5")
    resp = await base.chat_complete(messages, model=model)

    content = resp["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        return {"comms": {"raw": str(content)}}

    if not isinstance(parsed, dict) or "comms" not in parsed:
        return {"comms": {"raw": str(content)}}
    return parsed