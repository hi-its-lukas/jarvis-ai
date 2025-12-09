"""Helpers for filtering Home Assistant entities based on power state."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from app.logic.device_classifier import DeviceClassifier, DeviceDomain, EntityState, default_classifier


def _coerce_entity(state: dict[str, Any] | EntityState) -> EntityState:
    if isinstance(state, EntityState):
        return state
    return EntityState(
        entity_id=str(state.get("entity_id", "")),
        state=str(state.get("state", "unknown")),
        attributes=state.get("attributes") or {},
    )


def get_devices_on(
    states: Iterable[dict[str, Any] | EntityState],
    *,
    domains: Sequence[str] = (DeviceDomain.LIGHT.value, DeviceDomain.SWITCH.value),
    classifier: DeviceClassifier | None = None,
) -> list[EntityState]:
    classifier = classifier or default_classifier()
    allowed_domains = {domain.lower() for domain in domains}

    active_entities: list[EntityState] = []
    for state in states:
        entity = _coerce_entity(state)
        domain = classifier.classify(entity.entity_id, entity.attributes)
        if domain not in allowed_domains:
            continue
        if not classifier.domain_supports_on_off(domain):
            continue
        if classifier.is_on_state(domain, entity.state):
            active_entities.append(entity)
    return active_entities


def get_lights_on(
    states: Iterable[dict[str, Any] | EntityState],
    *,
    classifier: DeviceClassifier | None = None,
) -> list[EntityState]:
    return get_devices_on(
        states, domains=(DeviceDomain.LIGHT.value,), classifier=classifier
    )


def get_switches_on(
    states: Iterable[dict[str, Any] | EntityState],
    *,
    classifier: DeviceClassifier | None = None,
) -> list[EntityState]:
    return get_devices_on(
        states, domains=(DeviceDomain.SWITCH.value,), classifier=classifier
    )

