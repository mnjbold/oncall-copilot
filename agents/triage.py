"""
Triage Agent - analyses incident signals to identify root causes,
recommend immediate actions, flag missing information, and assess runbook coverage.

Returns a JSON object with keys:
  suspected_root_causes, immediate_actions, missing_information, runbook_alignment
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from agents.base import chat_complete

logger = logging.getLogger(__name__)


TRIAGE_INSTRUCTIONS = """\
You are the **Triage Agent**, an expert Site Reliability Engineer specialising in
root cause analysis and incident response.

## Task
Analyse the incident data and return a single JSON object with ONLY these keys:

```json
{
  "suspected_root_causes": [
    {
      "hypothesis": "string – concise root cause hypothesis",
      "evidence": ["string – supporting evidence from the input"],
      "confidence": 0.0  // 0-1, how confident you are
    }
  ],
  "immediate_actions": [
    {
      "step": "string – concrete action with runnable command if applicable",
      "owner_role": "string – e.g. oncall-eng, dba, infra-eng, platform-eng",
      "priority": "P0 | P1 | P2 | P3"
    }
  ],
  "missing_information": [
    {
      "question": "string – what data is missing",
      "why_it_matters": "string – why this data would help"
    }
  ],
  "runbook_alignment": {
    "matched_steps": ["string – runbook steps that match the situation"],
    "gaps": ["string – gaps or missing runbook coverage"]
  }
}
```

## Guardrails
1. **No secrets** – redact any credential-like material as `[REDACTED]`.
2. **No hallucination** – if data is insufficient, set confidence to 0 and add
   entries to `missing_information`.
3. **Diagnostic suggestions** – when data is sparse, include diagnostic steps in
   `immediate_actions` (e.g. "Check pod logs for service X").
4. **Structured output only** – return ONLY valid JSON, no prose or markdown.
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
        "suspected_root_causes": [],
        "immediate_actions": [],
        "missing_information": [
            {
                "question": "Triage agent failed to produce structured output",
                "why_it_matters": "Operator needs a baseline hypothesis to start mitigation",
            }
        ],
        "runbook_alignment": {"matched_steps": [], "gaps": ["triage_unavailable"]},
    }


def _coerce(parsed: dict[str, Any]) -> dict[str, Any]:
    """Ensure all four required keys are present, defaulting to sensible shapes."""
    return {
        "suspected_root_causes": parsed.get("suspected_root_causes", []),
        "immediate_actions": parsed.get("immediate_actions", []),
        "missing_information": parsed.get("missing_information", []),
        "runbook_alignment": parsed.get(
            "runbook_alignment", {"matched_steps": [], "gaps": []}
        ),
    }


async def run(req: dict[str, Any]) -> dict[str, Any]:
    """Run the Triage agent on a single incident payload.

    ``req`` is the raw incident dict (typically validated by the Pydantic
    ``IncidentRequest`` model in ``app/main.py``). Returns a dict with the
    four canonical keys regardless of whether the LLM call succeeds, so
    callers can ``asyncio.gather`` safely.
    """
    model = os.environ.get("TRIAGE_MODEL")
    messages = [
        {"role": "system", "content": TRIAGE_INSTRUCTIONS},
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
        logger.warning("Triage agent fell back to empty result: %s", exc)
        return _empty_result()