"""
Summary Agent – produces a concise incident summary.

Returns a JSON object with keys:
  summary (containing what_happened and current_status)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from agents.base import chat_complete

logger = logging.getLogger(__name__)


SUMMARY_INSTRUCTIONS = """\
You are the **Summary Agent**, an expert at distilling complex incident data
into clear, concise summaries for SRE teams.

## Task
Read the incident data and return a single JSON object with ONLY this key:

```json
{
  "summary": {
    "what_happened": "string – 2-4 sentence factual summary of the incident including affected services, failure mode, and scope",
    "current_status": "string – current state: ONGOING, MITIGATED, MONITORING, or RESOLVED with brief detail"
  }
}
```

## Guidelines
- **what_happened**: Lead with the trigger event and time. Include which services
  are affected and the failure mode. Be precise about impact scope.
- **current_status**: Use one of ONGOING / MITIGATED / MONITORING / RESOLVED as a
  prefix, followed by a brief detail of the current state.
- If the timeframe has an `end` timestamp, the incident is resolved.
- If no `end` timestamp, the incident is ongoing unless other signals say otherwise.
- **Structured output only** – return ONLY valid JSON, no prose or markdown.
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
        "summary": {
            "what_happened": "Summary agent failed to produce structured output",
            "current_status": "ONGOING – summary unavailable",
        }
    }


def _coerce(parsed: dict[str, Any]) -> dict[str, Any]:
    """Ensure the ``summary`` key is present and has the expected shape."""
    summary = parsed.get("summary")
    if not isinstance(summary, dict):
        # Model returned a string or other scalar — wrap it.
        return {
            "summary": {
                "what_happened": str(summary) if summary is not None else "",
                "current_status": "ONGOING",
            }
        }
    return {
        "summary": {
            "what_happened": summary.get("what_happened", ""),
            "current_status": summary.get("current_status", "ONGOING"),
        }
    }


async def run(req: dict[str, Any]) -> dict[str, Any]:
    """Run the Summary agent on a single incident payload.

    ``req`` is the raw incident dict (typically validated by the Pydantic
    ``IncidentRequest`` model in ``app/main.py``). Always returns
    ``{"summary": {...}}`` so callers can ``asyncio.gather`` safely.
    """
    model = os.environ.get("SUMMARY_MODEL", "gemini-2.5-flash")
    messages = [
        {"role": "system", "content": SUMMARY_INSTRUCTIONS},
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
        logger.warning("Summary agent fell back to empty result: %s", exc)
        return _empty_result()