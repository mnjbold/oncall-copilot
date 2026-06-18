"""FastAPI app for the on-call copilot."""
from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .orchestrator import handle_incident

app = FastAPI(title="On-Call Copilot", version="0.1.0")


class IncidentRequest(BaseModel):
    title: str
    service: str
    severity_hint: str | None = None
    context: str = ""
    logs: str | None = None
    metrics: str | None = None


class IncidentResponse(BaseModel):
    """Merged response from the 4 specialist agents.

    The four specialist agents return differently-shaped payloads, so the
    orchestrator flattens their top-level keys into a single dict. The
    fields below are the union of what each agent contributes on a normal
    run; per-agent error keys (``<agent>_error``) may also appear.
    """

    # Triage agent
    suspected_root_causes: list[dict[str, Any]] = Field(default_factory=list)
    immediate_actions: list[dict[str, Any]] = Field(default_factory=list)
    missing_information: list[dict[str, Any]] = Field(default_factory=list)
    runbook_alignment: dict[str, Any] = Field(default_factory=dict)
    # Summary agent
    summary: dict[str, Any] = Field(default_factory=dict)
    # Comms agent
    comms: dict[str, Any] = Field(default_factory=dict)
    # PIR agent
    post_incident_report: dict[str, Any] = Field(default_factory=dict)
    # Orchestrator meta
    elapsed_ms: int = 0


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/incident", response_model=IncidentResponse)
async def post_incident(req: IncidentRequest) -> IncidentResponse:
    """Dispatch an incident to the 4 specialist agents in parallel."""
    merged = await handle_incident(req.model_dump())
    return IncidentResponse(**merged)


def _port() -> int:
    return int(os.environ.get("PORT", "8080"))


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=_port())
