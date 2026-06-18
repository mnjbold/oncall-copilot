"""Tests for the PIR specialist agent.

The LLM call (``agents.pir.chat_complete``) is mocked at the
``agents.pir`` import site so tests never hit the HolySheep relay.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agents import pir
from agents.pir import (
    PIR_INSTRUCTIONS,
    _build_user_prompt,
    _coerce,
    _empty_result,
    _extract_message_content,
    run,
)


# ---------------------------------------------------------------------------
# Fixtures / canned payloads
# ---------------------------------------------------------------------------

CANONICAL_PAYLOAD: dict[str, Any] = {
    "title": "Checkout API 5xx spike",
    "service": "checkout-api",
    "severity_hint": "SEV2",
    "context": "Error rate climbed from 0.4% to 12% at 14:02 UTC after deploy v2.41.",
    "logs": "GET /checkout 502 - upstream connect timeout",
    "metrics": "p99 latency 4.2s, error_rate 12%",
}

EXPECTED_RESULT: dict[str, Any] = {
    "post_incident_report": {
        "timeline": [
            {"time": "13:55Z", "event": "Deploy v2.41 rolled out to production"},
            {"time": "14:02Z", "event": "Error rate climbed from 0.4% to 12%"},
            {"time": "14:18Z", "event": "Deploy v2.41 rolled back; error rate recovered"},
        ],
        "customer_impact": (
            "Approximately 12% of checkout requests failed for 16 minutes, "
            "blocking an estimated 4,200 customers from completing purchases."
        ),
        "prevention_actions": [
            "Add canary deploy stage (5% traffic for 10 min) before full rollout - owner: platform-eng",
            "Add automated rollback trigger when error_rate exceeds 5% for 3 minutes - owner: sre-oncall",
            "Capture full upstream connection pool metrics on checkout-api - owner: observability-eng",
        ],
    }
}


def _mock_response(content: Any) -> dict[str, Any]:
    """Build an OpenAI-style chat-completions envelope around ``content``."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "deepseek-v4-pro",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


# ---------------------------------------------------------------------------
# Static / constant checks
# ---------------------------------------------------------------------------


def test_pir_instructions_non_empty() -> None:
    assert PIR_INSTRUCTIONS.strip()
    assert "PIR Agent" in PIR_INSTRUCTIONS
    assert "post_incident_report" in PIR_INSTRUCTIONS
    assert "timeline" in PIR_INSTRUCTIONS
    assert "customer_impact" in PIR_INSTRUCTIONS
    assert "prevention_actions" in PIR_INSTRUCTIONS


def test_build_user_prompt_includes_core_fields() -> None:
    prompt = _build_user_prompt(CANONICAL_PAYLOAD)
    assert "Checkout API 5xx spike" in prompt
    assert "checkout-api" in prompt
    assert "SEV2" in prompt
    assert "deploy v2.41" in prompt
    assert "upstream connect timeout" in prompt
    assert "p99 latency" in prompt


def test_build_user_prompt_handles_minimal_payload() -> None:
    prompt = _build_user_prompt({"title": "Boom", "service": "svc"})
    assert "Boom" in prompt
    assert "svc" in prompt
    assert "(none)" in prompt  # severity_hint default
    # Logs / metrics sections should be absent when not supplied
    assert "Logs:" not in prompt
    assert "Metrics:" not in prompt


def test_extract_message_content_happy_path() -> None:
    payload = _mock_response("hello")
    assert _extract_message_content(payload) == "hello"


def test_extract_message_content_raises_on_bad_envelope() -> None:
    with pytest.raises(ValueError):
        _extract_message_content({"choices": []})


def test_empty_result_shape() -> None:
    result = _empty_result()
    assert set(result.keys()) == {"post_incident_report"}
    assert isinstance(result["post_incident_report"], dict)
    assert result["post_incident_report"]["timeline"] == []
    assert result["post_incident_report"]["prevention_actions"] == []
    assert result["post_incident_report"]["customer_impact"]


def test_coerce_passes_through_valid_dict() -> None:
    out = _coerce(EXPECTED_RESULT)
    assert out == EXPECTED_RESULT


def test_coerce_fills_missing_fields() -> None:
    out = _coerce({"post_incident_report": {"customer_impact": "users blocked"}})
    assert out["post_incident_report"]["customer_impact"] == "users blocked"
    assert out["post_incident_report"]["timeline"] == []
    assert out["post_incident_report"]["prevention_actions"] == []


def test_coerce_wraps_scalar_report() -> None:
    out = _coerce({"post_incident_report": "just a string"})
    assert out["post_incident_report"]["timeline"] == []
    assert out["post_incident_report"]["customer_impact"] == "just a string"
    assert out["post_incident_report"]["prevention_actions"] == []


def test_coerce_handles_missing_pir_key() -> None:
    out = _coerce({"foo": "bar"})
    assert set(out.keys()) == {"post_incident_report"}
    assert out["post_incident_report"]["timeline"] == []
    assert out["post_incident_report"]["customer_impact"]


def test_coerce_coerces_non_list_timeline() -> None:
    out = _coerce(
        {"post_incident_report": {"timeline": "oops", "customer_impact": "x"}}
    )
    assert out["post_incident_report"]["timeline"] == []
    assert out["post_incident_report"]["customer_impact"] == "x"


def test_coerce_coerces_non_list_prevention_actions() -> None:
    out = _coerce({"post_incident_report": {"prevention_actions": "oops"}})
    assert out["post_incident_report"]["prevention_actions"] == []


# ---------------------------------------------------------------------------
# run() with mocked chat_complete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_canonical_shape_on_valid_response() -> None:
    mock_chat = AsyncMock(return_value=_mock_response(json.dumps(EXPECTED_RESULT)))
    with patch.object(pir, "chat_complete", mock_chat):
        result = await run(CANONICAL_PAYLOAD)

    assert set(result.keys()) == {"post_incident_report"}
    assert isinstance(result["post_incident_report"]["timeline"], list)
    assert len(result["post_incident_report"]["timeline"]) == 3
    assert result["post_incident_report"]["timeline"][0]["time"] == "13:55Z"
    assert "12%" in result["post_incident_report"]["customer_impact"]
    assert len(result["post_incident_report"]["prevention_actions"]) == 3


@pytest.mark.asyncio
async def test_run_passes_pir_model_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PIR_MODEL", "deepseek-v4-pro")
    mock_chat = AsyncMock(return_value=_mock_response(json.dumps(EXPECTED_RESULT)))
    with patch.object(pir, "chat_complete", mock_chat) as patched:
        await run(CANONICAL_PAYLOAD)

    assert patched.call_count == 1
    args, kwargs = patched.call_args
    assert kwargs["model"] == "deepseek-v4-pro"
    messages = args[0]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == PIR_INSTRUCTIONS
    assert messages[1]["role"] == "user"
    assert "Checkout API 5xx spike" in messages[1]["content"]


@pytest.mark.asyncio
async def test_run_uses_default_model_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PIR_MODEL", raising=False)

    captured: dict[str, Any] = {}

    async def fake_chat(messages, *, model=None, timeout=30.0):
        captured["model"] = model
        return _mock_response(json.dumps(EXPECTED_RESULT))

    with patch.object(pir, "chat_complete", fake_chat):
        await run(CANONICAL_PAYLOAD)

    assert captured["model"] == "deepseek-v4-pro"


@pytest.mark.asyncio
async def test_run_coerces_partial_response() -> None:
    """LLM returns only ``customer_impact`` - agent fills in the rest."""
    partial = {"post_incident_report": {"customer_impact": "users blocked"}}
    mock_chat = AsyncMock(return_value=_mock_response(json.dumps(partial)))
    with patch.object(pir, "chat_complete", mock_chat):
        result = await run(CANONICAL_PAYLOAD)

    assert result["post_incident_report"]["customer_impact"] == "users blocked"
    assert result["post_incident_report"]["timeline"] == []
    assert result["post_incident_report"]["prevention_actions"] == []


@pytest.mark.asyncio
async def test_run_falls_back_on_invalid_json() -> None:
    mock_chat = AsyncMock(return_value=_mock_response("not json at all"))
    with patch.object(pir, "chat_complete", mock_chat):
        result = await run(CANONICAL_PAYLOAD)

    fallback = _empty_result()
    assert result["post_incident_report"]["customer_impact"] == (
        fallback["post_incident_report"]["customer_impact"]
    )
    assert (
        result["post_incident_report"]["timeline"]
        == fallback["post_incident_report"]["timeline"]
    )
    assert result["post_incident_report"]["prevention_actions"] == (
        fallback["post_incident_report"]["prevention_actions"]
    )


@pytest.mark.asyncio
async def test_run_falls_back_on_non_object_json() -> None:
    mock_chat = AsyncMock(return_value=_mock_response(json.dumps([1, 2, 3])))
    with patch.object(pir, "chat_complete", mock_chat):
        result = await run(CANONICAL_PAYLOAD)
    fallback = _empty_result()
    assert result["post_incident_report"]["customer_impact"] == (
        fallback["post_incident_report"]["customer_impact"]
    )


@pytest.mark.asyncio
async def test_run_falls_back_on_chat_complete_exception() -> None:
    mock_chat = AsyncMock(side_effect=RuntimeError("upstream timeout"))
    with patch.object(pir, "chat_complete", mock_chat):
        result = await run(CANONICAL_PAYLOAD)
    fallback = _empty_result()
    assert result["post_incident_report"]["customer_impact"] == (
        fallback["post_incident_report"]["customer_impact"]
    )
    assert (
        result["post_incident_report"]["timeline"]
        == fallback["post_incident_report"]["timeline"]
    )


@pytest.mark.asyncio
async def test_run_falls_back_on_malformed_envelope() -> None:
    mock_chat = AsyncMock(return_value={"no_choices_key": True})
    with patch.object(pir, "chat_complete", mock_chat):
        result = await run(CANONICAL_PAYLOAD)
    fallback = _empty_result()
    assert result["post_incident_report"]["customer_impact"] == (
        fallback["post_incident_report"]["customer_impact"]
    )


@pytest.mark.asyncio
async def test_run_works_with_minimal_request() -> None:
    """req with only title+service should not crash."""
    mock_chat = AsyncMock(return_value=_mock_response(json.dumps(EXPECTED_RESULT)))
    with patch.object(pir, "chat_complete", mock_chat) as patched:
        result = await run({"title": "T", "service": "S"})
    assert result["post_incident_report"]["timeline"][0]["time"] == "13:55Z"
    # And the user prompt should still be a string with the values
    user_msg = patched.call_args[0][0][1]["content"]
    assert "T" in user_msg and "S" in user_msg