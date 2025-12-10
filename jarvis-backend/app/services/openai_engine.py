"""OpenAI-compatible chat completion engine for Jarvis."""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional

from app.services.entity_service import EntityService
from app.services.function_registry import FUNCTION_DEFINITIONS
from app.services.ollama_api import OllamaAPI
from app.services.discovery import DiscoveryService


class OpenAICompatibleEngine:
    """Generate OpenAI-compatible responses backed by Jarvis services."""

    def __init__(
        self,
        *,
        discovery: DiscoveryService | None = None,
        ollama: OllamaAPI | None = None,
        entity_service: EntityService | None = None,
    ) -> None:
        self._discovery = discovery
        self._ollama = ollama
        self._entities = entity_service or EntityService(discovery=discovery)

    async def generate_chat_completion(
        self,
        *,
        messages: List[dict],
        functions: Optional[List[dict]] = None,
        function_call: Optional[str | dict] = "auto",
        entities: Optional[List[dict]] = None,
        model: str | None = None,
    ) -> Dict[str, Any]:
        if not messages:
            raise ValueError("No messages provided")

        user_message = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        if not user_message or not user_message.get("content"):
            raise ValueError("User message missing content")

        content = str(user_message.get("content"))
        lowered = content.lower()

        available_entities = list(entities or [])
        if not available_entities:
            available_entities = self._load_entities_from_discovery()

        if function_call == "none":
            return self._build_text_response("Jarvis hat deine Anfrage erhalten.", model=model)

        # Questions about current state
        if self._is_status_query(lowered):
            return self._handle_status_question(lowered, available_entities, model)

        # Direct control intents
        command_response = self._handle_direct_command(lowered, available_entities, model)
        if command_response:
            return command_response

        # LLM fallback
        llm_choice = await self._llm_route(content, functions or FUNCTION_DEFINITIONS)
        if llm_choice:
            return llm_choice

        return self._build_text_response(
            "Ich habe deine Anfrage erhalten, konnte aber keine passende Aktion ableiten.",
            model=model,
        )

    def _load_entities_from_discovery(self) -> List[dict]:
        if not self._discovery:
            return []
        records: Iterable[Any] = getattr(self._discovery, "_entities", [])
        return [
            {
                "entity_id": record.entity_id,
                "name": record.name,
                "friendly_name": record.attributes.get("friendly_name", record.name),
                "state": record.state,
                "attributes": record.attributes,
            }
            for record in records
        ]

    def _handle_status_question(
        self, lowered: str, entities: List[dict], model: str | None
    ) -> Dict[str, Any]:
        keywords = {"welche", "was", "status", "list"}
        if not any(word in lowered for word in keywords):
            return self._build_text_response("Ich habe deine Anfrage erhalten.", model=model)

        lights_on = [
            entity
            for entity in entities
            if self._entities.classify_entity(entity.get("entity_id", ""), entity.get("attributes", {}))
            == "light"
            and str(entity.get("state", "")).lower() == "on"
        ]
        if lights_on:
            names = ", ".join(entity.get("friendly_name") or entity.get("name") or entity.get("entity_id") for entity in lights_on)
            return self._build_text_response(
                f"Diese Lichter sind eingeschaltet: {names}.", model=model
            )
        return self._build_text_response("Aktuell sind keine Lichter eingeschaltet.", model=model)

    def _handle_direct_command(
        self, lowered: str, entities: List[dict], model: str | None
    ) -> Optional[Dict[str, Any]]:
        is_turn_on = any(word in lowered for word in ["schalte", "mach", "turn on", "einschalten", "anmachen"])
        is_turn_off = "aus" in lowered or "turn off" in lowered or "ausschalten" in lowered
        wants_cover = any(word in lowered for word in ["rollo", "rollladen", "jalousie"])
        wants_brightness = bool(re.search(r"\b\d{1,3}%", lowered)) or "helligkeit" in lowered
        wants_temperature = "temperatur" in lowered or "heizen" in lowered

        target_entity = self._match_entity(lowered, entities)
        if not target_entity:
            return None

        entity_id = target_entity.get("entity_id")
        domain = self._entities.classify_entity(entity_id or "", target_entity.get("attributes", {}))

        if wants_cover and domain == "cover":
            if any(word in lowered for word in ["runter", "schließ", "zu", "close"]):
                return self._function_call("close_cover", {"entity_id": entity_id}, model)
            return self._function_call("open_cover", {"entity_id": entity_id}, model)

        if wants_temperature and domain in {"climate", "sensor"}:
            temperature_value = self._extract_number(lowered)
            if temperature_value is not None:
                return self._function_call(
                    "set_temperature", {"entity_id": entity_id, "temperature": temperature_value}, model
                )

        if domain == "light":
            if wants_brightness:
                brightness = self._extract_percentage(lowered) or 128
                return self._function_call(
                    "set_brightness", {"entity_id": entity_id, "brightness": brightness}, model
                )
            if is_turn_off:
                return self._function_call("turn_off", {"entity_id": entity_id}, model)
            if is_turn_on:
                return self._function_call("turn_on", {"entity_id": entity_id}, model)

        if domain == "media_player" and ("pause" in lowered or "play" in lowered):
            return self._function_call("media_play_pause", {"entity_id": entity_id}, model)

        if is_turn_on and domain in {"switch", "fan"}:
            return self._function_call("turn_on", {"entity_id": entity_id}, model)
        if is_turn_off and domain in {"switch", "fan"}:
            return self._function_call("turn_off", {"entity_id": entity_id}, model)

        return None

    def _match_entity(self, lowered: str, entities: List[dict]) -> Optional[dict]:
        keywords = [word for word in re.split(r"\W+", lowered) if word and len(word) > 2]
        for entity in entities:
            name = str(entity.get("friendly_name") or entity.get("name") or "").lower()
            entity_id = str(entity.get("entity_id", "")).lower()
            if any(keyword in name or keyword in entity_id for keyword in keywords):
                return entity
        return None

    async def _llm_route(self, content: str, functions: List[dict]) -> Optional[Dict[str, Any]]:
        if not self._ollama:
            return None

        system_instruction = (
            "Du konvertierst Smart-Home-Anfragen in JSON-Funktionsaufrufe."
            " Nutze ausschließlich die bereitgestellten Funktionen."
        )
        prompt = json.dumps(
            {
                "messages": content,
                "functions": functions,
            },
            ensure_ascii=False,
        )
        try:
            result = await self._ollama.ask_ollama(
                prompt=prompt,
                system_instruction=system_instruction,
                use_smart_model=True,
            )
        except Exception:
            return None

        name = result.get("function") or result.get("tool_name")
        arguments = result.get("arguments") or {}
        if not name:
            return None

        return self._function_call(name, arguments, None)

    def _function_call(self, name: str, arguments: Dict[str, Any], model: str | None) -> Dict[str, Any]:
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model or "jarvis",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "function_call": {
                            "name": name,
                            "arguments": json.dumps(arguments, ensure_ascii=False),
                        },
                    },
                    "finish_reason": "function_call",
                }
            ],
        }

    def _build_text_response(self, text: str, model: str | None = None) -> Dict[str, Any]:
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model or "jarvis",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": str(text)},
                    "finish_reason": "stop",
                }
            ],
        }

    def _extract_percentage(self, lowered: str) -> Optional[int]:
        match = re.search(r"(\d{1,3})%", lowered)
        if not match:
            return None
        value = int(match.group(1))
        return max(1, min(value, 100))

    def _extract_number(self, lowered: str) -> Optional[float]:
        numbers = re.findall(r"(\d{1,2}(?:[\.,]\d)?)", lowered)
        for num in numbers:
            try:
                return float(num.replace(",", "."))
            except ValueError:
                continue
        return None

    def _is_status_query(self, lowered: str) -> bool:
        return any(word in lowered for word in ["welche", "was", "status", "sind an", "sind aus"])

