\"\"\"Shared helper for HolySheep LLM calls (stub for the scaffold).\"\"\"
from __future__ import annotations

import os
from typing import Any

import httpx


async def chat_complete(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    \"\"\"Send a chat-completions request to the HolySheep relay.

    Implementation lives in a follow-up dispatch. Kept here so agents/
    imports resolve during scaffold smoke-tests.
    \"\"\"
    base_url = os.environ.get(\"HOLYSHEEP_BASE_URL\", \"https://api.holysheep.ai/v1\")
    api_key = os.environ.get(\"HOLYSHEEP_API_KEY\", \"\")
    chosen_model = model or os.environ.get(\"TRIAGE_MODEL\", \"\")
    headers = {\"Authorization\": f\"Bearer {api_key}\"} if api_key else {}
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f\"{base_url}/chat/completions\",
            headers=headers,
            json={\"model\": chosen_model, \"messages\": messages},
        )
        resp.raise_for_status()
        return resp.json()
