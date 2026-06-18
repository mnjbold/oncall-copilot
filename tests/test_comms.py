"""Tests for the Comms agent."""

from __future__ import annotations

import json

import pytest

from agents import comms
from agents import base


class _FakeResponse(dict):
    """Minimal stand-in for an OpenAI-style chat completion response."""


def _make_response(content: str) -> dict:
    return _FakeResponse(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ]
        }
    )


@pytest.mark.asyncio
async def test_run_returns_parsed_comms(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "slack_update": ":rotating_light: *INC-1234* SEV2 — checkout 5xx spike. "
        "Investigating. Next update 14:30 UTC.",
        "stakeholder_update": "Customers may see checkout failures. "
        "Engineering is mitigating; next update at 14:30 UTC.",
    }
    captured: dict = {}

    async def fake_chat_complete(messages, *, model=None, timeout=30.0):  # noqa: ANN001
        captured["messages"] = messages
        captured["model"] = model
        return _make_response(json.dumps({"comms": payload}))

    monkeypatch.setattr(base, "chat_complete", fake_chat_complete)

    req = {
        "title": "Checkout API 5xx spike",
        "service": "checkout-api",
        "severity_hint": "SEV2",
        "context": "Error rate climbed from 0.4% to 12% at 14:02 UTC.",
    }
    result = await comms.run(req)

    assert result == {"comms": payload}
    # System prompt must carry the verbatim instructions.
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][0]["content"] == comms.COMMS_INSTRUCTIONS
    # User payload must be the JSON-serialised request.
    user_content = captured["messages"][1]["content"]
    assert json.loads(user_content) == req
    # Model must default to claude-haiku-4-5 (or env override).
    assert captured["model"] == "claude-haiku-4-5"


@pytest.mark.asyncio
async def test_run_honours_comms_model_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("COMMS_MODEL", "claude-haiku-4-5")
    captured: dict = {}

    async def fake_chat_complete(messages, *, model=None, timeout=30.0):  # noqa: ANN001
        captured["model"] = model
        return _make_response(json.dumps({"comms": {"slack_update": "x", "stakeholder_update": "y"}}))

    monkeypatch.setattr(base, "chat_complete", fake_chat_complete)
    await comms.run({"title": "t", "service": "s"})
    assert captured["model"] == "claude-haiku-4-5"


@pytest.mark.asyncio
async def test_run_falls_back_on_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_chat_complete(messages, *, model=None, timeout=30.0):  # noqa: ANN001
        return _make_response("not json at all")

    monkeypatch.setattr(base, "chat_complete", fake_chat_complete)
    result = await comms.run({"title": "t", "service": "s"})
    assert "comms" in result
    assert "raw" in result["comms"]


@pytest.mark.asyncio
async def test_run_falls_back_when_comms_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_chat_complete(messages, *, model=None, timeout=30.0):  # noqa: ANN001
        return _make_response(json.dumps({"foo": "bar"}))

    monkeypatch.setattr(base, "chat_complete", fake_chat_complete)
    result = await comms.run({"title": "t", "service": "s"})
    assert "comms" in result
    assert result["comms"]["raw"] == '{"foo": "bar"}'


def test_comms_instructions_verbatim() -> None:
    """The instructions must match the upstream leestott reference verbatim."""
    expected = """\
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
    assert comms.COMMS_INSTRUCTIONS == expected