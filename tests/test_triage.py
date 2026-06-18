"""Tests for the Triage specialist agent.

The LLM call (``agents.triage.chat_complete``) is mocked at the
``agents.triage`` import site so tests never hit the HolySheep relay.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agents import triage
from agents.triage import (
    TRIAGE_INSTRUCTIONS,
    _build_user_prompt,
    _empty_result,
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
    "suspected_root_causes": [
        {
            "hypothesis": "Bad deploy v2.41 introduced upstream connection pool exhaustion",
            "evidence": ["deploy v2.41", "upstream connect timeout"],
            "confidence": 0.82,
        },
        {
            "hypothesis": "Database connection saturation",
            "evidence": ["5xx spike"],
            "confidence": 0.15,
        },
    ],
    "immediate_actions": [
        {
            "step": "Roll back deploy v2.41",
            "owner_role": "platform-eng",
            "priority": "P0",
        },
        {
            "step": "Capture heap dump from checkout-api pods",
            "owner_role": "oncall-eng",
            "priority": "P1",
        },
    ],
    "missing_information": [
        {
            "question": "What changed in v2.41?",
            "why_it_matters": "Confirms deploy-related hypothesis",
        }
    ],
    "runbook_alignment": {
        "matched_steps": ["rollback procedure", "paging oncall-eng"],
        "gaps": ["no documented rollback verification step"],
    },
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


def test_triage_instructions_non_empty() -> None:
    assert TRIAGE_INSTRUCTIONS.strip()
    assert "Triage Agent" in TRIAGE_INSTRUCTIONS
    assert "suspected_root_causes" in TRIAGE_INSTRUCTIONS
    assert "immediate_actions" in TRIAGE_INSTRUCTIONS
    assert "missing_information" in TRIAGE_INSTRUCTIONS
    assert "runbook_alignment" in TRIAGE_INSTRUCTIONS


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


def test_empty_result_shape() -> None:
    result = _empty_result()
    assert set(result.keys()) == {
        "suspected_root_causes",
        "immediate_actions",
        "missing_information",
        "runbook_alignment",
    }
    assert result["suspected_root_causes"] == []
    assert result["immediate_actions"] == []
    assert result["missing_information"]  # carries a fallback message
    assert result["runbook_alignment"]["gaps"] == ["triage_unavailable"]


# ---------------------------------------------------------------------------
# run() with mocked chat_complete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_canonical_keys_on_valid_response() -> None:
    mock_chat = AsyncMock(
        return_value=_mock_response(json.dumps(EXPECTED_RESULT))
    )
    with patch.object(triage, "chat_complete", mock_chat):
        result = await run(CANONICAL_PAYLOAD)

    assert set(result.keys()) == {
        "suspected_root_causes",
        "immediate_actions",
        "missing_information",
        "runbook_alignment",
    }
    assert result["suspected_root_causes"][0]["hypothesis"].startswith("Bad deploy")
    assert result["immediate_actions"][0]["priority"] == "P0"
    assert result["runbook_alignment"]["matched_steps"] == [
        "rollback procedure",
        "paging oncall-eng",
    ]


@pytest.mark.asyncio
async def test_run_passes_triage_model_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRIAGE_MODEL", "deepseek-v4-pro")
    mock_chat = AsyncMock(return_value=_mock_response(json.dumps(EXPECTED_RESULT)))
    with patch.object(triage, "chat_complete", mock_chat) as patched:
        await run(CANONICAL_PAYLOAD)

    assert patched.call_count == 1
    args, kwargs = patched.call_args
    assert kwargs["model"] == "deepseek-v4-pro"
    messages = args[0]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == TRIAGE_INSTRUCTIONS
    assert messages[1]["role"] == "user"
    assert "Checkout API 5xx spike" in messages[1]["content"]


@pytest.mark.asyncio
async def test_run_coerces_partial_response() -> None:
    """LLM returns only some keys - agent fills in defaults."""
    partial = {
        "suspected_root_causes": [
            {"hypothesis": "x", "evidence": [], "confidence": 0.5}
        ]
    }
    mock_chat = AsyncMock(return_value=_mock_response(json.dumps(partial)))
    with patch.object(triage, "chat_complete", mock_chat):
        result = await run(CANONICAL_PAYLOAD)

    assert len(result["suspected_root_causes"]) == 1
    assert result["immediate_actions"] == []
    assert result["missing_information"] == []
    assert result["runbook_alignment"] == {"matched_steps": [], "gaps": []}


@pytest.mark.asyncio
async def test_run_falls_back_on_invalid_json() -> None:
    mock_chat = AsyncMock(return_value=_mock_response("not json at all"))
    with patch.object(triage, "chat_complete", mock_chat):
        result = await run(CANONICAL_PAYLOAD)

    fallback = _empty_result()
    assert result["suspected_root_causes"] == fallback["suspected_root_causes"]
    assert result["immediate_actions"] == fallback["immediate_actions"]
    assert result["runbook_alignment"]["gaps"] == fallback["runbook_alignment"]["gaps"]
    # The fallback carries a sentinel in missing_information
    assert result["missing_information"][0]["question"] == (
        "Triage agent failed to produce structured output"
    )


@pytest.mark.asyncio
async def test_run_falls_back_on_non_object_json() -> None:
    mock_chat = AsyncMock(return_value=_mock_response(json.dumps([1, 2, 3])))
    with patch.object(triage, "chat_complete", mock_chat):
        result = await run(CANONICAL_PAYLOAD)
    assert result["suspected_root_causes"] == []
    assert result["missing_information"][0]["question"].startswith("Triage agent")


@pytest.mark.asyncio
async def test_run_falls_back_on_chat_complete_exception() -> None:
    mock_chat = AsyncMock(side_effect=RuntimeError("upstream timeout"))
    with patch.object(triage, "chat_complete", mock_chat):
        result = await run(CANONICAL_PAYLOAD)
    assert result["suspected_root_causes"] == []
    assert result["runbook_alignment"]["gaps"] == ["triage_unavailable"]


@pytest.mark.asyncio
async def test_run_falls_back_on_malformed_envelope() -> None:
    mock_chat = AsyncMock(return_value={"no_choices_key": True})
    with patch.object(triage, "chat_complete", mock_chat):
        result = await run(CANONICAL_PAYLOAD)
    assert result["suspected_root_causes"] == []
    assert result["missing_information"][0]["question"].startswith("Triage agent")


@pytest.mark.asyncio
async def test_run_works_with_minimal_request() -> None:
    """req with only title+service should not crash."""
    mock_chat = AsyncMock(return_value=_mock_response(json.dumps(EXPECTED_RESULT)))
    with patch.object(triage, "chat_complete", mock_chat) as patched:
        result = await run({"title": "T", "service": "S"})
    assert result["suspected_root_causes"][0]["hypothesis"].startswith("Bad deploy")
    # And the user prompt should still be a string with the values
    user_msg = patched.call_args[0][0][1]["content"]
    assert "T" in user_msg and "S" in user_msg