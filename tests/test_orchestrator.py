"""Tests for the orchestrator (``app.orchestrator.handle_incident``).

The four specialist ``run`` functions are mocked at the orchestrator's
import site (``app.orchestrator.{triage,summary,comms,pir}.run``) so no
LLM traffic is generated.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from app import orchestrator
from app.orchestrator import handle_incident


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

TRIAGE_RESULT: dict[str, Any] = {
    "suspected_root_causes": [
        {
            "hypothesis": "Bad deploy v2.41",
            "evidence": ["upstream connect timeout"],
            "confidence": 0.82,
        }
    ],
    "immediate_actions": [
        {"step": "Roll back deploy v2.41", "owner_role": "platform-eng", "priority": "P0"}
    ],
    "missing_information": [],
    "runbook_alignment": {"matched_steps": [], "gaps": []},
}

SUMMARY_RESULT: dict[str, Any] = {
    "summary": {
        "what_happened": "Checkout 5xx spiked to 12% at 14:02 UTC after deploy v2.41.",
        "current_status": "MITIGATED – rolled back to v2.40.",
    }
}

COMMS_RESULT: dict[str, Any] = {
    "comms": {
        "slack_update": ":rotating_light: *INC-1234* SEV2 — checkout 5xx spike.",
        "stakeholder_update": "Customers may see checkout failures; engineering is mitigating.",
    }
}

PIR_RESULT: dict[str, Any] = {
    "post_incident_report": {
        "timeline": [{"time": "14:02Z", "event": "5xx spike detected"}],
        "customer_impact": "Approx 4,200 customers affected for 16 minutes.",
        "prevention_actions": ["Add canary stage for checkout deploys"],
    }
}


def _make_async(return_value: Any) -> AsyncMock:
    """Build an ``AsyncMock`` that returns ``return_value`` when awaited."""
    mock = AsyncMock(return_value=return_value)
    return mock


# ---------------------------------------------------------------------------
# Happy path: all 4 agents called and merged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_incident_calls_all_four_agents() -> None:
    with (
        patch.object(orchestrator.triage, "run", _make_async(TRIAGE_RESULT)) as triage_run,
        patch.object(orchestrator.summary, "run", _make_async(SUMMARY_RESULT)) as summary_run,
        patch.object(orchestrator.comms, "run", _make_async(COMMS_RESULT)) as comms_run,
        patch.object(orchestrator.pir, "run", _make_async(PIR_RESULT)) as pir_run,
    ):
        result = await handle_incident(CANONICAL_PAYLOAD)

    assert triage_run.await_count == 1
    assert summary_run.await_count == 1
    assert comms_run.await_count == 1
    assert pir_run.await_count == 1

    # Each agent receives the canonical payload (or a copy of it).
    for mocked in (triage_run, summary_run, comms_run, pir_run):
        args, _kwargs = mocked.call_args
        # First positional arg is the payload.
        assert args[0] == CANONICAL_PAYLOAD


@pytest.mark.asyncio
async def test_handle_incident_returns_merged_payload_with_elapsed_ms() -> None:
    with (
        patch.object(orchestrator.triage, "run", _make_async(TRIAGE_RESULT)),
        patch.object(orchestrator.summary, "run", _make_async(SUMMARY_RESULT)),
        patch.object(orchestrator.comms, "run", _make_async(COMMS_RESULT)),
        patch.object(orchestrator.pir, "run", _make_async(PIR_RESULT)),
    ):
        result = await handle_incident(CANONICAL_PAYLOAD)

    # Triage fields
    assert result["suspected_root_causes"] == TRIAGE_RESULT["suspected_root_causes"]
    assert result["immediate_actions"] == TRIAGE_RESULT["immediate_actions"]
    assert result["runbook_alignment"] == TRIAGE_RESULT["runbook_alignment"]

    # Summary field (dict, not string)
    assert result["summary"] == SUMMARY_RESULT["summary"]
    assert isinstance(result["summary"], dict)

    # Comms field
    assert result["comms"] == COMMS_RESULT["comms"]

    # PIR field
    assert result["post_incident_report"] == PIR_RESULT["post_incident_report"]

    # Orchestrator meta
    assert "elapsed_ms" in result
    assert isinstance(result["elapsed_ms"], int)
    assert result["elapsed_ms"] >= 0


@pytest.mark.asyncio
async def test_handle_incident_runs_agents_concurrently() -> None:
    """Each agent takes ~0.1s; total elapsed should be ~0.1s, not ~0.4s."""

    async def slow_but_not_too_slow(_: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(0.1)
        return {"ok": True}

    with (
        patch.object(orchestrator.triage, "run", side_effect=slow_but_not_too_slow),
        patch.object(orchestrator.summary, "run", side_effect=slow_but_not_too_slow),
        patch.object(orchestrator.comms, "run", side_effect=slow_but_not_too_slow),
        patch.object(orchestrator.pir, "run", side_effect=slow_but_not_too_slow),
    ):
        result = await handle_incident(CANONICAL_PAYLOAD)

    # If agents were serial, this would be ~400ms+. Allow generous slack.
    assert result["elapsed_ms"] < 350, (
        f"Agents appear to be running serially: elapsed_ms={result['elapsed_ms']}"
    )


@pytest.mark.asyncio
async def test_handle_incident_accepts_pydantic_like_object() -> None:
    """``handle_incident`` should accept anything exposing ``model_dump``."""

    class _Req:
        def model_dump(self) -> dict[str, Any]:
            return dict(CANONICAL_PAYLOAD)

    with (
        patch.object(orchestrator.triage, "run", _make_async(TRIAGE_RESULT)) as triage_run,
        patch.object(orchestrator.summary, "run", _make_async(SUMMARY_RESULT)),
        patch.object(orchestrator.comms, "run", _make_async(COMMS_RESULT)),
        patch.object(orchestrator.pir, "run", _make_async(PIR_RESULT)),
    ):
        await handle_incident(_Req())

    # Payload was coerced to dict and forwarded as-is to each agent.
    args, _ = triage_run.call_args
    assert args[0] == CANONICAL_PAYLOAD


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_incident_timeout_per_agent() -> None:
    """A slow agent should be cut off and produce a ``<name>_error: timeout`` key."""

    async def slow(_: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(5.0)
        return {"never": "reached"}

    with (
        patch.object(orchestrator.triage, "run", side_effect=slow),
        patch.object(orchestrator.summary, "run", _make_async(SUMMARY_RESULT)),
        patch.object(orchestrator.comms, "run", _make_async(COMMS_RESULT)),
        patch.object(orchestrator.pir, "run", _make_async(PIR_RESULT)),
    ):
        result = await handle_incident(CANONICAL_PAYLOAD, timeout=0.1)

    # The slow agent (triage) was timed out.
    assert result["triage_error"] == "timeout"
    assert result["triage_timeout_s"] == pytest.approx(0.1)

    # The other agents returned normally — their dicts were merged in.
    assert result["summary"] == SUMMARY_RESULT["summary"]
    assert result["comms"] == COMMS_RESULT["comms"]
    assert result["post_incident_report"] == PIR_RESULT["post_incident_report"]
    assert "elapsed_ms" in result


@pytest.mark.asyncio
async def test_handle_incident_does_not_raise_on_agent_exception() -> None:
    """An agent that raises should be reported as ``<name>_error`` and not abort the call."""

    async def boom(_: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("LLM relay unreachable")

    with (
        patch.object(orchestrator.triage, "run", side_effect=boom),
        patch.object(orchestrator.summary, "run", _make_async(SUMMARY_RESULT)),
        patch.object(orchestrator.comms, "run", _make_async(COMMS_RESULT)),
        patch.object(orchestrator.pir, "run", _make_async(PIR_RESULT)),
    ):
        result = await handle_incident(CANONICAL_PAYLOAD)

    assert "triage_error" in result
    assert "RuntimeError" in result["triage_error"]
    assert "LLM relay unreachable" in result["triage_error"]

    # Healthy agents are still present in the merged response.
    assert result["summary"] == SUMMARY_RESULT["summary"]
    assert result["comms"] == COMMS_RESULT["comms"]
    assert result["post_incident_report"] == PIR_RESULT["post_incident_report"]


@pytest.mark.asyncio
async def test_default_timeout_is_thirty_seconds() -> None:
    """The module-level default must be 30s per CLAUDE.md."""
    assert orchestrator.DEFAULT_TIMEOUT_S == 30.0
