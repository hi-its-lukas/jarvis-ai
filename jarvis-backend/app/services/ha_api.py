"""Asynchronous Home Assistant REST API wrapper."""

from __future__ import annotations

from typing import Any

import httpx


class HomeAssistantAPI:
    """Minimal async client for interacting with Home Assistant's REST API."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_seconds: int = 10,
    ) -> None:
        base_url_str = ""
        if base_url:
            try:
                base_url_str = str(base_url).rstrip("/")
            except Exception:
                base_url_str = ""

        self._base_url = base_url_str
        self._token = token
        self._timeout = timeout_seconds

    async def call_service(self, domain: str, service: str, data: dict[str, Any]) -> Any:
        """Invoke a Home Assistant service with sanitized payload."""

        url = f"{self._base_url}/api/services/{domain}/{service}"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, headers=headers, json=data)
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise RuntimeError("Home Assistant service call timed out") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Home Assistant service call failed: {exc}") from exc
        return response.json()

    async def fetch_states(self) -> list[dict[str, Any]]:
        """Retrieve the full list of entity states."""

        url = f"{self._base_url}/api/states"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise RuntimeError("Home Assistant states fetch timed out") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Home Assistant states fetch failed: {exc}") from exc
        return response.json()
