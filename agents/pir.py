"""
PIR Agent - Post-Incident Report specialist. Constructs timeline,
assesses customer impact, and recommends prevention actions.

Returns a JSON object with keys:
  post_incident_report (containing timeline, customer_impact, prevention_actions)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from agents.base import chat_complete

logger = logging.getLogger(__name__)


PIR_INSTRUCTIONS = """\
You are the **PIR Agent**, an expert post-incident report writer for SRE teams.

## Task
Read the incident data and return a single JSON object with ONLY this key:

```json
{
  "post_incident_report": {
    "timeline": [
      {"time": "HH:MMZ or ISO timestamp", "event": "string – what happened"}
    ],
    "customer_impact": "string – clear statement of how customers were affected, including scope and duration",
    "prevention_actions": [
      "string – specific, actionable prevention measure with owner suggestion"
    ]
  }
}
```

## Guidelines
- **Timeline**: Reconstruct from alerts, logs, and metrics timestamps. Order
  chronologically. Use the earliest signal as the start. If the incident is
  ongoing, end with `{"time": "ONGOING", "event": "..."}`.
- **Customer impact**: Quantify where possible (users affected, % error rate,
  revenue estimate). If the incident had no customer impact, say so explicitly.
- **Prevention actions**: Be specific and actionable. Include technical changes,
  process improvements, and monitoring enhancements. Suggest owners by role.
- **Structured output only** - return ONLY valid JSON, no prose or markdown.
"""


def _build_user_prompt(req: dict[str, Any]) -> str:
    """Render the incident request as a compact user message for the LLM."""
    title = req.get("title", "(no title)")
    service = req.get("service", "(no service)")
    severity_hint = req.get("severity_hint", "(none)")
    context = req.get("context", "")

    parts = [
        f"Title: {title}",
        f"Service: {service}",
        f"Severity hint: {severity_hint}",
    ]
    if context:
        parts.append(f"Context:\n{context}")
    if req.get("logs"):
        parts.append(f"Logs:\n{req['logs']}")
    if req.get("metrics"):
        parts.append(f"Metrics:\n{req['metrics']}")
    return "\n\n".join(parts)


def _extract_message_content(response: dict[str, Any]) -> str:
    """Pull the assistant text out of an OpenAI-style chat-completions payload."""
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected chat-completions payload shape: {exc}") from exc


def _empty_result() -> dict[str, Any]:
    """Fallback structure returned on parse / transport errors."""
    return {
        "post_incident_report": {
            "timeline": [],
            "customer_impact": "PIR agent failed to produce structured output",
            "prevention_actions": [],
        }
    }


def _coerce(parsed: dict[str, Any]) -> dict[str, Any]:
    """Ensure the ``post_incident_report`` key is present with the expected shape."""
    pir = parsed.get("post_incident_report")
    if not isinstance(pir, dict):
        return {
            "post_incident_report": {
                "timeline": [],
                "customer_impact": (
                    str(pir) if pir is not None
                    else "PIR agent returned no report payload"
                ),
                "prevention_actions": [],
            }
        }
    timeline = pir.get("timeline", [])
    if not isinstance(timeline, list):
        timeline = []
    prevention_actions = pir.get("prevention_actions", [])
    if not isinstance(prevention_actions, list):
        prevention_actions = []
    return {
        "post_incident_report": {
            "timeline": timeline,
            "customer_impact": pir.get(
                "customer_impact",
                "PIR agent did not return a customer_impact statement",
            ),
            "prevention_actions": prevention_actions,
        }
    }


async def run(req: dict[str, Any]) -> dict[str, Any]:
    """Run the PIR agent on a single incident payload.

    ``req`` is the raw incident dict (typically validated by the Pydantic
    ``IncidentRequest`` model in ``app/main.py``). Always returns
    ``{"post_incident_report": {...}}`` so callers can ``asyncio.gather`` safely.
    """
    model = os.environ.get("PIR_MODEL", "deepseek-v4-pro")
    messages = [
        {"role": "system", "content": PIR_INSTRUCTIONS},
        {"role": "user", "content": _build_user_prompt(req)},
    ]

    try:
        response = await chat_complete(messages, model=model)
        content = _extract_message_content(response)
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("LLM returned non-object JSON")
        return _coerce(parsed)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully per CLAUDE.md
        logger.warning("PIR agent fell back to empty result: %s", exc)
        return _empty_result()