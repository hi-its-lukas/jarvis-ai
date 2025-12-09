"""Adapter to expose a simplified OpenAI-compatible interface for Home Assistant."""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Iterable, List, Optional

from app.services.entity_classifier import classify_entity


def get_entities_by_state(entities: Iterable[dict], state: str) -> list[dict]:
    """Return entities that match the desired state (case-insensitive)."""

    desired_state = (state or "").lower()
    return [entity for entity in entities if str(entity.get("state", "")).lower() == desired_state]


def _build_base_response() -> dict:
    return {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
    }


def _format_choice(content: Optional[str] = None, function_call: Optional[dict] = None, finish_reason: str = "stop") -> dict:
    return {
        "index": 0,
        "message": {
            "role": "assistant",
            "content": content,
            **({"function_call": function_call} if function_call else {}),
        },
        "finish_reason": finish_reason,
    }


def _stringify_arguments(arguments: Dict[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False)


def _build_function_call_response(name: str, arguments: Dict[str, Any]) -> dict:
    response = _build_base_response()
    response["choices"] = [
        _format_choice(
            content=None,
            function_call={"name": name, "arguments": _stringify_arguments(arguments)},
            finish_reason="function_call",
        )
    ]
    return response


def _build_text_response(content: str) -> dict:
    response = _build_base_response()
    response["choices"] = [_format_choice(content=content, finish_reason="stop")]
    return response


def turn_on(entity_id: str) -> dict:
    return _build_function_call_response("turn_on", {"entity_id": entity_id})


def turn_off(entity_id: str) -> dict:
    return _build_function_call_response("turn_off", {"entity_id": entity_id})


def set_brightness(entity_id: str, brightness: int) -> dict:
    return _build_function_call_response("set_brightness", {"entity_id": entity_id, "brightness": brightness})


def set_temperature(entity_id: str, temperature: float) -> dict:
    return _build_function_call_response("set_temperature", {"entity_id": entity_id, "temperature": temperature})


def media_play_pause(entity_id: str) -> dict:
    return _build_function_call_response("media_play_pause", {"entity_id": entity_id})


def set_volume(entity_id: str, volume: float) -> dict:
    return _build_function_call_response("set_volume", {"entity_id": entity_id, "volume": volume})


def open_cover(entity_id: str) -> dict:
    return _build_function_call_response("open_cover", {"entity_id": entity_id})


def close_cover(entity_id: str) -> dict:
    return _build_function_call_response("close_cover", {"entity_id": entity_id})


def set_color(entity_id: str, rgb_color: list[int]) -> dict:
    return _build_function_call_response("set_color", {"entity_id": entity_id, "rgb_color": rgb_color})


def _entity_matches_keywords(entity: dict, keywords: List[str]) -> bool:
    name = str(entity.get("name", "")).lower()
    entity_id = str(entity.get("entity_id", "")).lower()
    identifier = entity_id.split(".")[-1].replace("_", " ")

    return any(keyword in name or keyword in identifier for keyword in keywords if keyword)


def _find_entity(entities: Iterable[dict], keywords: List[str], domain: Optional[str] = None) -> Optional[dict]:
    for entity in entities:
        if domain and classify_entity(entity.get("entity_id", ""), entity.get("attributes", {})) != domain:
            continue
        if _entity_matches_keywords(entity, keywords):
            return entity
    return None


async def generate_ha_response(messages: List[dict], functions: Optional[List[dict]], entities: List[dict]) -> dict:
    """Generate an OpenAI-compatible response for Home Assistant queries."""

    user_message = next((msg for msg in reversed(messages) if msg.get("role") == "user"), None)
    content = str(user_message.get("content", "")) if user_message else ""
    lowered = content.lower()

    # Query about lights that are currently on
    if ("welche" in lowered or "was" in lowered) and any(word in lowered for word in ["lampen", "lichter", "licht"]):
        on_lights = [
            entity
            for entity in get_entities_by_state(entities, "on")
            if classify_entity(entity.get("entity_id", ""), entity.get("attributes", {})) == "light"
        ]
        if on_lights:
            names = ", ".join(entity.get("name", entity.get("entity_id")) for entity in on_lights)
            return _build_text_response(f"Diese Lichter sind eingeschaltet: {names}.")
        return _build_text_response("Aktuell sind keine Lichter eingeschaltet.")

    # Temperature questions
    if any(keyword in lowered for keyword in ["temperatur", "warm"]):
        keywords = [word for word in lowered.replace("?", "").split() if len(word) > 2]
        target_entity = _find_entity(
            entities,
            keywords,
            domain=None,
        )
        if target_entity and classify_entity(target_entity.get("entity_id", ""), target_entity.get("attributes", {})) in [
            "climate",
            "sensor",
        ]:
            temperature = target_entity.get("state", "unbekannt")
            name = target_entity.get("name", target_entity.get("entity_id"))
            return _build_text_response(f"Die Temperatur von {name} beträgt {temperature}.")

    # Cover control
    if any(keyword in lowered for keyword in ["rollo", "rolladen", "rollos", "jalousie"]):
        keywords = [
            word
            for word in lowered.replace("?", "").split()
            if word not in {"mach", "die", "den", "das", "bitte"} and len(word) > 2
        ]
        entity = _find_entity(entities, keywords, domain="cover")
        if entity:
            target_action = "close_cover" if any(word in lowered for word in ["runter", "schließ", "zu"]) else "open_cover"
            return close_cover(entity.get("entity_id")) if target_action == "close_cover" else open_cover(entity.get("entity_id"))

    # Light control
    if any(keyword in lowered for keyword in ["schalte", "mach", "dimme", "erhelle"]):
        keywords = [
            word
            for word in lowered.replace("?", "").split()
            if word not in {"schalte", "mach", "das", "die", "den", "bitte", "ein", "auf"} and len(word) > 2
        ]
        entity = _find_entity(entities, keywords, domain="light")
        if entity:
            if "aus" in lowered:
                return turn_off(entity.get("entity_id"))
            if any(word in lowered for word in ["heller", "dunkler", "dimme", "helligkeit"]):
                return set_brightness(entity.get("entity_id"), 128)
            return turn_on(entity.get("entity_id"))

    # Fallback
    return _build_text_response("Ich habe deine Anfrage erhalten, konnte aber keine passende Aktion ableiten.")

