"""Main orchestration logic for Jarvis backend."""

from __future__ import annotations

import re
from typing import Any

from app.core.security import SanitizationError
from app.logic.context import ConversationContext
from app.logic.device_classifier import classify_entity
from app.services import tool_registry
from app.services.discovery import DiscoveryService
from app.services.ha_api import HomeAssistantAPI
from app.services.ollama_api import OllamaAPI
from app.services.semantic_router import semantic_route

_NEGATION_KEYWORDS = {
    "aus",
    "abschalten",
    "ausschalten",
    "ausmachen",
    "stop",
    "stoppe",
    "pause",
    "off",
    "schließen",
    "schliesse",
    "close",
    "halt",
}
_BRIGHTNESS_PATTERN = re.compile(r"(\d{1,3})\s*%")


class Processor:
    def __init__(
        self,
        *,
        ha_api: HomeAssistantAPI,
        discovery: DiscoveryService,
        ollama: OllamaAPI,
        context: ConversationContext,
    ) -> None:
        self._ha_api = ha_api
        self._discovery = discovery
        self._ollama = ollama
        self._context = context

    async def process(self, text: str) -> dict[str, Any]:
        route = semantic_route(text)
        if route:
            direct_result = await self._handle_direct_path(text, route.intent, route.domain)
            if direct_result is not None:
                return direct_result
        return await self._handle_llm_path(text)

    async def _handle_direct_path(
        self,
        text: str,
        intent: str,
        domain: str,
    ) -> dict[str, Any] | None:
        turn_off = _contains_keyword(text, _NEGATION_KEYWORDS)
        tool = tool_registry.tool_for_intent(intent, turn_off=turn_off)
        if not tool:
            return None

        matches = await self._discovery.search(text, limit=1, domains=(domain,))
        if not matches:
            return None
        entity = matches[0]

        args: dict[str, Any] = {"entity_id": entity.entity_id}
        if tool.name == "ha_light_turn_on":
            brightness = _extract_brightness_pct(text)
            if brightness is not None:
                args["brightness_pct"] = brightness
        if tool.name == "ha_climate_set_temperature":
            temperature = _extract_temperature(text)
            if temperature is not None:
                args["temperature"] = temperature
            else:
                return None
        for required in tool.required_args:
            if required not in args:
                return None

        try:
            domain_name, service_name, payload = tool_registry.prepare_service_payload(
                tool.name, args
            )
        except (SanitizationError, ValueError) as exc:
            raise RuntimeError(f"Invalid direct command arguments: {exc}") from exc

        result = await self._ha_api.call_service(domain_name, service_name, payload)

        entity_domain = classify_entity(entity.entity_id, entity.attributes)
        summary = f"{service_name} ({entity_domain}) -> {entity.entity_id}"
        self._context.add_entry(text, summary)
        return {
            "path": "direct",
            "entity_id": entity.entity_id,
            "service": f"{domain_name}.{service_name}",
            "result": result,
        }

    async def _handle_llm_path(self, text: str) -> dict[str, Any]:
        context_entities = self._discovery.get_context_entities(limit=5)
        context_block = self._context.as_prompt_block()
        system_instruction = (
            "You convert German or English smart home requests into Home Assistant tool calls.\n"
            "Always respond with JSON: {\"tool_name\": string, \"arguments\": object}.\n"
            "Never explain yourself. Only return the JSON object.\n"
            "Available tools:\n"
            f"{tool_registry.describe_tools_for_prompt()}"
        )
        user_prompt = (
            f"User request: {text}\n"
            f"Known entities: {context_entities}\n"
            f"{context_block}"
        )

        llm_response = await self._ollama.ask_ollama(
            prompt=user_prompt,
            system_instruction=system_instruction,
        )
        tool_name = llm_response.get("tool_name") or ""
        arguments = llm_response.get("arguments") or {}
        if not tool_name:
            raise RuntimeError("LLM response missing tool_name")

        domain_name, service_name, payload = tool_registry.prepare_service_payload(
            tool_name,
            arguments,
        )
        result = await self._ha_api.call_service(domain_name, service_name, payload)
        summary = f"{service_name} -> {payload.get('entity_id', 'unknown')}"
        self._context.add_entry(text, summary)
        return {
            "path": "llm",
            "service": f"{domain_name}.{service_name}",
            "entity_id": payload.get("entity_id"),
            "payload": payload,
            "result": result,
        }


def _contains_keyword(text: str, keywords: set[str]) -> bool:
    normalized = re.sub(r"[^\wäöüß]+", " ", text.lower())
    wrapped = f" {normalized} "
    return any(f" {keyword} " in wrapped for keyword in keywords)


def _extract_brightness_pct(text: str) -> int | None:
    match = _BRIGHTNESS_PATTERN.search(text)
    if not match:
        return None
    value = int(match.group(1))
    return max(1, min(value, 100))


def _extract_temperature(text: str) -> float | None:
    numbers = re.findall(r"(\d{1,2}(?:\.\d)?)", text)
    for num in numbers:
        value = float(num.replace(",", "."))
        if 5 <= value <= 35:
            return value
    return None
