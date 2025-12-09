"""Entity discovery and fuzzy search utilities."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import monotonic
from typing import Any, Sequence

from rapidfuzz import fuzz, process

from app.logic.device_classifier import DeviceClassifier, DeviceDomain, default_classifier
from .ha_api import HomeAssistantAPI

IGNORED_DOMAINS = {
    DeviceDomain.DEVICE_TRACKER.value,
    DeviceDomain.BINARY_SENSOR.value,
    DeviceDomain.SENSOR.value,
    "person",
    "zone",
    "sun",
}


@dataclass(slots=True)
class EntityRecord:
    entity_id: str
    name: str
    domain: str
    state: str
    attributes: dict[str, Any]

    @property
    def search_text(self) -> str:
        alias = self.attributes.get("friendly_name") or self.name
        return f"{alias} {self.entity_id.replace('_', ' ')}"


class DiscoveryService:
    def __init__(
        self,
        ha_api: HomeAssistantAPI,
        *,
        cache_max_entities: int = 200,
        min_score: int = 65,
        classifier: DeviceClassifier | None = None,
    ) -> None:
        self._ha_api = ha_api
        self._cache_max_entities = cache_max_entities
        self._min_score = min_score
        self._entities: list[EntityRecord] = []
        self._last_refresh = 0.0
        self._classifier = classifier or default_classifier()

    async def refresh(self) -> None:
        """Fetch latest entity states and cache in memory."""

        states = await self._ha_api.fetch_states()
        filtered: list[EntityRecord] = []
        for state in states:
            entity_id = state.get("entity_id")
            if not entity_id:
                continue
            attributes = state.get("attributes") or {}
            domain = self._classifier.classify(entity_id, attributes)
            if domain in IGNORED_DOMAINS:
                continue
            name = attributes.get("friendly_name") or entity_id
            record = EntityRecord(
                entity_id=entity_id,
                name=str(name),
                domain=domain,
                state=str(state.get("state", "unknown")),
                attributes=attributes,
            )
            filtered.append(record)
            if len(filtered) >= self._cache_max_entities:
                break

        self._entities = filtered
        self._last_refresh = monotonic()

    def last_refresh_age(self) -> float:
        return max(0.0, monotonic() - self._last_refresh)

    async def search(
        self,
        query: str,
        *,
        limit: int = 1,
        domains: Sequence[str] | None = None,
    ) -> list[EntityRecord]:
        """Return entities ordered by fuzzy similarity to the query."""

        if not query or not self._entities:
            return []

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._search_sync, query, limit, domains
        )

    def _search_sync(
        self,
        query: str,
        limit: int,
        domains: Sequence[str] | None = None,
    ) -> list[EntityRecord]:
        if domains:
            domain_set = {domain.lower() for domain in domains}
            records = [
                record for record in self._entities if record.domain.lower() in domain_set
            ]
        else:
            records = list(self._entities)

        choices: dict[str, str] = {
            record.entity_id: record.search_text for record in records
        }
        if not choices:
            return []

        matches = process.extract(
            query,
            choices,
            scorer=fuzz.WRatio,
            limit=limit,
        )
        results: list[EntityRecord] = []
        for entity_id, score, _ in matches:
            if score < self._min_score:
                continue
            record = next((rec for rec in records if rec.entity_id == entity_id), None)
            if record:
                results.append(record)
        return results

    def get_context_entities(
        self,
        *,
        limit: int = 5,
        domains: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return simplified entity descriptions for LLM context."""

        selected: list[EntityRecord] = []
        for record in self._entities:
            if domains and record.domain not in domains:
                continue
            selected.append(record)
            if len(selected) >= limit:
                break

        context: list[dict[str, Any]] = []
        for record in selected:
            context.append(
                {
                    "entity_id": record.entity_id,
                    "domain": record.domain,
                    "state": record.state,
                    "friendly_name": record.attributes.get("friendly_name", record.name),
                }
            )
        return context
