"""Tests for the Summary specialist agent.

The LLM call (``agents.summary.chat_complete``) is mocked at the
``agents.summary`` import site so tests never hit the HolySheep relay.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agents import summary
from agents.summary import (
    SUMMARY_INSTRUCTIONS,
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
    "summary": {
        "what_happened": (
            "Checkout API 5xx rate spiked to 12% at 14:02 UTC after deploy v2.41; "
            "upstream connect timeouts affecting all checkout traffic."
        ),
        "current_status": "MITIGATED – rolled back to v2.40, error rate recovering.",
    }
}


def _mock_response(content: Any) -> dict[str, Any]:
    """Build an OpenAI-style chat-completions envelope around ``content``."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "gemini-2.5-flash",
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


def test_summary_instructions_non_empty() -> None:
    assert SUMMARY_INSTRUCTIONS.strip()
    assert "Summary Agent" in SUMMARY_INSTRUCTIONS
    assert "what_happened" in SUMMARY_INSTRUCTIONS
    assert "current_status" in SUMMARY_INSTRUCTIONS


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
    assert set(result.keys()) == {"summary"}
    assert isinstance(result["summary"], dict)
    assert result["summary"]["current_status"].startswith("ONGOING")
    assert result["summary"]["what_happened"]


def test_coerce_passes_through_valid_dict() -> None:
    out = _coerce(EXPECTED_RESULT)
    assert out == EXPECTED_RESULT


def test_coerce_fills_missing_fields() -> None:
    out = _coerce({"summary": {"what_happened": "boom"}})
    assert out["summary"]["what_happened"] == "boom"
    assert out["summary"]["current_status"] == "ONGOING"


def test_coerce_wraps_scalar_summary() -> None:
    out = _coerce({"summary": "just a string"})
    assert out["summary"]["what_happened"] == "just a string"
    assert out["summary"]["current_status"] == "ONGOING"


def test_coerce_handles_missing_summary_key() -> None:
    out = _coerce({"foo": "bar"})
    assert out == {"summary": {"what_happened": "", "current_status": "ONGOING"}}


# ---------------------------------------------------------------------------
# run() with mocked chat_complete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_canonical_shape_on_valid_response() -> None:
    mock_chat = AsyncMock(return_value=_mock_response(json.dumps(EXPECTED_RESULT)))
    with patch.object(summary, "chat_complete", mock_chat):
        result = await run(CANONICAL_PAYLOAD)

    assert set(result.keys()) == {"summary"}
    assert "what_happened" in result["summary"]
    assert "current_status" in result["summary"]
    assert result["summary"]["what_happened"].startswith("Checkout API")
    assert result["summary"]["current_status"].startswith("MITIGATED")


@pytest.mark.asyncio
async def test_run_passes_summary_model_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SUMMARY_MODEL", "gemini-2.5-flash")
    mock_chat = AsyncMock(return_value=_mock_response(json.dumps(EXPECTED_RESULT)))
    with patch.object(summary, "chat_complete", mock_chat) as patched:
        await run(CANONICAL_PAYLOAD)

    assert patched.call_count == 1
    args, kwargs = patched.call_args
    assert kwargs["model"] == "gemini-2.5-flash"
    messages = args[0]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SUMMARY_INSTRUCTIONS
    assert messages[1]["role"] == "user"
    assert "Checkout API 5xx spike" in messages[1]["content"]


@pytest.mark.asyncio
async def test_run_uses_default_model_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SUMMARY_MODEL", raising=False)

    captured: dict[str, Any] = {}

    async def fake_chat(messages, *, model=None, timeout=30.0):
        captured["model"] = model
        return _mock_response(json.dumps(EXPECTED_RESULT))

    with patch.object(summary, "chat_complete", fake_chat):
        await run(CANONICAL_PAYLOAD)

    assert captured["model"] == "gemini-2.5-flash"


@pytest.mark.asyncio
async def test_run_coerces_partial_response() -> None:
    """LLM returns only ``what_happened`` - agent fills in the rest."""
    partial = {"summary": {"what_happened": "boom"}}
    mock_chat = AsyncMock(return_value=_mock_response(json.dumps(partial)))
    with patch.object(summary, "chat_complete", mock_chat):
        result = await run(CANONICAL_PAYLOAD)

    assert result["summary"]["what_happened"] == "boom"
    assert result["summary"]["current_status"] == "ONGOING"


@pytest.mark.asyncio
async def test_run_falls_back_on_invalid_json() -> None:
    mock_chat = AsyncMock(return_value=_mock_response("not json at all"))
    with patch.object(summary, "chat_complete", mock_chat):
        result = await run(CANONICAL_PAYLOAD)

    fallback = _empty_result()
    assert result["summary"]["what_happened"] == fallback["summary"]["what_happened"]
    assert result["summary"]["current_status"] == fallback["summary"]["current_status"]


@pytest.mark.asyncio
async def test_run_falls_back_on_non_object_json() -> None:
    mock_chat = AsyncMock(return_value=_mock_response(json.dumps([1, 2, 3])))
    with patch.object(summary, "chat_complete", mock_chat):
        result = await run(CANONICAL_PAYLOAD)
    # Falls back to the empty-result sentinel because the LLM payload
    # was a JSON array rather than an object.
    fallback = _empty_result()
    assert result["summary"]["what_happened"] == fallback["summary"]["what_happened"]
    assert result["summary"]["current_status"] == fallback["summary"]["current_status"]


@pytest.mark.asyncio
async def test_run_falls_back_on_chat_complete_exception() -> None:
    mock_chat = AsyncMock(side_effect=RuntimeError("upstream timeout"))
    with patch.object(summary, "chat_complete", mock_chat):
        result = await run(CANONICAL_PAYLOAD)
    fallback = _empty_result()
    assert result["summary"]["what_happened"] == fallback["summary"]["what_happened"]
    assert result["summary"]["current_status"] == fallback["summary"]["current_status"]


@pytest.mark.asyncio
async def test_run_falls_back_on_malformed_envelope() -> None:
    mock_chat = AsyncMock(return_value={"no_choices_key": True})
    with patch.object(summary, "chat_complete", mock_chat):
        result = await run(CANONICAL_PAYLOAD)
    fallback = _empty_result()
    assert result["summary"]["what_happened"] == fallback["summary"]["what_happened"]
    assert result["summary"]["current_status"] == fallback["summary"]["current_status"]


@pytest.mark.asyncio
async def test_run_works_with_minimal_request() -> None:
    """req with only title+service should not crash."""
    mock_chat = AsyncMock(return_value=_mock_response(json.dumps(EXPECTED_RESULT)))
    with patch.object(summary, "chat_complete", mock_chat) as patched:
        result = await run({"title": "T", "service": "S"})
    assert result["summary"]["what_happened"].startswith("Checkout API")
    # And the user prompt should still be a string with the values
    user_msg = patched.call_args[0][0][1]["content"]
    assert "T" in user_msg and "S" in user_msg