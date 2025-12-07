"""Async wrapper for interacting with a local Ollama instance."""

from __future__ import annotations

import json
from typing import Any

import httpx


class OllamaAPI:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_seconds: int = 60,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds

    async def ask_ollama(
        self,
        *,
        prompt: str,
        system_instruction: str,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """Send a structured prompt to Ollama using JSON mode."""

        payload = {
            "model": self._model,
            "format": "json",
            "stream": False,
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
        }
        url = f"{self._base_url}/api/chat"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise RuntimeError("Ollama request timed out") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        data = response.json()
        message_content = data.get("message", {}).get("content")
        if not message_content:
            raise RuntimeError("Ollama returned an empty response")

        try:
            return json.loads(message_content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama response was not valid JSON") from exc
