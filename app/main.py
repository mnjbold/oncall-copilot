\"\"\"FastAPI app for the on-call copilot (scaffold — agents stubbed).\"\"\"
from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title=\"On-Call Copilot\", version=\"0.1.0\")


class IncidentRequest(BaseModel):
    title: str
    service: str
    severity_hint: str | None = None
    context: str = \"\"
    logs: str | None = None
    metrics: str | None = None


class IncidentResponse(BaseModel):
    triage: dict[str, Any]
    summary: str
    comms: dict[str, Any]
    pir: dict[str, Any]


@app.get(\"/healthz\")
async def healthz() -> dict[str, str]:
    return {\"status\": \"ok\"}


@app.post(\"/incident\", response_model=IncidentResponse)
async def handle_incident(req: IncidentRequest) -> IncidentResponse:
    # TODO: dispatch to the 4 specialist agents via asyncio.gather()
    # against the HolySheep relay. Implementation lands in a follow-up
    # dispatch — this scaffold only validates wiring.
    raise HTTPException(status_code=501, detail=\"Agents not yet implemented\")


def _port() -> int:
    return int(os.environ.get(\"PORT\", \"8080\"))


if __name__ == \"__main__\":  # pragma: no cover
    import uvicorn

    uvicorn.run(\"app.main:app\", host=\"0.0.0.0\", port=_port())
