"""Async wrapper for interacting with a local Ollama instance."""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.core.config import get_settings

settings = get_settings()


class OllamaAPI:
    def __init__(
        self,
        base_url: str,
        model: str | None = None,
        *,
        timeout_seconds: int | None = None,
    ) -> None:
        base_url_str = ""
        if base_url:
            try:
                base_url_str = str(base_url).rstrip("/")
            except Exception:
                base_url_str = ""

        self._base_url = base_url_str
        self._default_model = model
        self._default_timeout = timeout_seconds

    async def ask_ollama(
        self,
        *,
        prompt: str,
        system_instruction: str,
        temperature: float = 0.1,
        use_smart_model: bool = False,
    ) -> dict[str, Any]:
        """Send a structured prompt to Ollama using JSON mode."""

        if use_smart_model:
            model = settings.ollama_model_smart or self._default_model
            timeout = settings.llm_timeout_smart_seconds or self._default_timeout
        else:
            model = settings.ollama_model_fast or self._default_model
            timeout = settings.llm_timeout_fast_seconds or self._default_timeout

        if not model:
            raise RuntimeError("No Ollama model configured")

        if not timeout:
            timeout = settings.llm_timeout_seconds

        print(f"[LLM] Using model: {model}")

        payload = {
            "model": model,
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
            async with httpx.AsyncClient(timeout=timeout) as client:
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
