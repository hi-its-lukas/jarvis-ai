"""Helpers for categorising Home Assistant entities."""

from __future__ import annotations

from typing import Any

_DOMAIN_ALIASES: dict[str, str] = {
    "lamp": "light",
    "licht": "light",
    "switch": "switch",
    "plug": "switch",
    "cover": "cover",
    "jalousie": "cover",
    "speaker": "media_player",
    "tv": "media_player",
}


def classify_entity(entity_id: str, attributes: dict[str, Any]) -> str:
    """Infer an entity's logical domain using metadata and id hints."""

    if not entity_id:
        return "unknown"

    domain = entity_id.split(".", maxsplit=1)[0]
    if domain in {"light", "switch", "cover", "media_player", "climate"}:
        return domain

    device_class = attributes.get("device_class", "").lower()
    if device_class in _DOMAIN_ALIASES:
        return _DOMAIN_ALIASES[device_class]

    for alias, mapped in _DOMAIN_ALIASES.items():
        if alias in entity_id.lower():
            return mapped

    return domain
