"""Helpers for Home Assistant entity classification and lookup."""

from __future__ import annotations

from typing import Dict, Iterable, List

from app.services.discovery import DiscoveryService, EntityRecord


class EntityService:
    """Provide simple entity lookups for OpenAI-compatible endpoints."""

    _DOMAIN_MAPPING = {
        "light": "light",
        "switch": "switch",
        "cover": "cover",
        "media_player": "media_player",
        "fan": "fan",
        "climate": "climate",
        "sensor": "sensor",
        "binary_sensor": "binary_sensor",
    }

    def __init__(self, discovery: DiscoveryService | None = None) -> None:
        self._discovery = discovery

    def classify_entity(self, entity_id: str, attributes: dict) -> str:
        domain = (entity_id or "").split(".")[0]
        return self._DOMAIN_MAPPING.get(domain, "unknown")

    def find_entities_by_name(self, name: str) -> List[Dict[str, str]]:
        if not self._discovery or not name:
            return []
        matches = self._discovery._search_sync(name, limit=5, domains=None)  # type: ignore[attr-defined]
        return [self._record_to_dict(record) for record in matches]

    def find_entities_by_state(self, state: str) -> List[Dict[str, str]]:
        if not self._discovery or not state:
            return []
        desired_state = state.lower()
        records: Iterable[EntityRecord] = getattr(self._discovery, "_entities", [])
        return [
            self._record_to_dict(record)
            for record in records
            if str(record.state).lower() == desired_state
        ]

    def _record_to_dict(self, record: EntityRecord) -> Dict[str, str]:
        return {
            "entity_id": record.entity_id,
            "name": record.name,
            "friendly_name": record.attributes.get("friendly_name", record.name),
            "state": record.state,
            "attributes": record.attributes,
        }

