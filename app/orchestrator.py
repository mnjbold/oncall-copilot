"""Orchestrator — runs the 4 specialist agents concurrently per incident.

Wraps ``triage``, ``summary``, ``comms``, and ``pir`` in ``asyncio.gather``
with per-agent timeouts (default 30s) and merges the per-agent dicts into a
single flat response, plus an ``elapsed_ms`` timing field.

Per CLAUDE.md:
- All four agents MUST be dispatched concurrently.
- Each agent call should be wrapped with a sensible timeout and degrade
  gracefully (return a partial / fallback dict).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from agents import triage, summary, comms, pir

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S: float = 30.0


def _coerce_payload(req: Any) -> dict[str, Any]:
    """Coerce a Pydantic model or mapping into a plain ``dict``."""
    if hasattr(req, "model_dump"):
        return req.model_dump()
    return dict(req)


async def _run_with_timeout(
    name: str,
    fn: Any,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """Run one agent under a timeout, logging start/end and per-agent duration.

    On timeout or any other failure, returns a small fallback dict (rather
    than raising) so ``asyncio.gather`` always resolves and the merged
    response stays structurally complete.
    """
    t0 = time.monotonic()
    logger.info(
        "orchestrator: starting %s agent (timeout=%.1fs)", name, timeout,
    )
    try:
        result = await asyncio.wait_for(fn(payload), timeout=timeout)
    except asyncio.TimeoutError:
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.warning(
            "orchestrator: %s agent timed out after %.1fs (%.1fms elapsed)",
            name, timeout, elapsed_ms,
        )
        return {
            f"{name}_error": "timeout",
            f"{name}_timeout_s": timeout,
        }
    except Exception as exc:  # noqa: BLE001 - degrade gracefully per CLAUDE.md
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.exception(
            "orchestrator: %s agent failed after %.1fms: %s",
            name, elapsed_ms, exc,
        )
        return {f"{name}_error": f"{type(exc).__name__}: {exc}"}

    elapsed_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "orchestrator: %s agent completed in %.1fms", name, elapsed_ms,
    )
    if not isinstance(result, dict):
        # Defensive: agents promise a dict, but if one ever returns a coroutine
        # or other object, wrap it rather than blowing up the merge.
        return {f"{name}_result": result}
    return result


async def handle_incident(
    req: Any,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Run all 4 specialist agents concurrently and return merged result.

    Args:
        req: Incident payload. Accepts a ``dict`` or any Pydantic model
            exposing ``model_dump`` (e.g. ``IncidentRequest``).
        timeout: Per-agent timeout in seconds. Defaults to 30s.

    Returns:
        A flat dict containing the merged keys from all four agents plus an
        ``elapsed_ms`` integer. Agent-level failures are reported as
        ``<agent>_error`` keys so the caller can surface partial results.
    """
    payload = _coerce_payload(req)
    t_start = time.monotonic()
    logger.info(
        "orchestrator: dispatching incident (title=%r, service=%r, timeout=%.1fs)",
        payload.get("title"), payload.get("service"), timeout,
    )

    triage_result, summary_result, comms_result, pir_result = await asyncio.gather(
        _run_with_timeout("triage", triage.run, payload, timeout),
        _run_with_timeout("summary", summary.run, payload, timeout),
        _run_with_timeout("comms", comms.run, payload, timeout),
        _run_with_timeout("pir", pir.run, payload, timeout),
    )

    elapsed_ms = int((time.monotonic() - t_start) * 1000)
    logger.info(
        "orchestrator: all agents finished in %dms (title=%r)",
        elapsed_ms, payload.get("title"),
    )

    return {
        **triage_result,
        **summary_result,
        **comms_result,
        **pir_result,
        "elapsed_ms": elapsed_ms,
    }
