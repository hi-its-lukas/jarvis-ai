"""Device domain classification helpers for Home Assistant entities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping


class DeviceDomain(StrEnum):
    """Known Home Assistant domains relevant for control and filtering."""

    ALARM_CONTROL_PANEL = "alarm_control_panel"
    BINARY_SENSOR = "binary_sensor"
    CAMERA = "camera"
    CLIMATE = "climate"
    COVER = "cover"
    DEVICE_TRACKER = "device_tracker"
    FAN = "fan"
    HUMIDIFIER = "humidifier"
    LIGHT = "light"
    LOCK = "lock"
    MEDIA_PLAYER = "media_player"
    SENSOR = "sensor"
    SWITCH = "switch"
    VACUUM = "vacuum"
    UNKNOWN = "unknown"


_DEFAULT_ALIASES: dict[str, DeviceDomain] = {
    "lamp": DeviceDomain.LIGHT,
    "lampe": DeviceDomain.LIGHT,
    "licht": DeviceDomain.LIGHT,
    "light": DeviceDomain.LIGHT,
    "switch": DeviceDomain.SWITCH,
    "plug": DeviceDomain.SWITCH,
    "outlet": DeviceDomain.SWITCH,
    "socket": DeviceDomain.SWITCH,
    "cover": DeviceDomain.COVER,
    "shade": DeviceDomain.COVER,
    "blind": DeviceDomain.COVER,
    "jalousie": DeviceDomain.COVER,
    "roller": DeviceDomain.COVER,
    "fan": DeviceDomain.FAN,
    "ventilator": DeviceDomain.FAN,
    "media": DeviceDomain.MEDIA_PLAYER,
    "speaker": DeviceDomain.MEDIA_PLAYER,
    "tv": DeviceDomain.MEDIA_PLAYER,
    "thermostat": DeviceDomain.CLIMATE,
    "ac": DeviceDomain.CLIMATE,
    "vacuum": DeviceDomain.VACUUM,
}


@dataclass(slots=True)
class EntityState:
    """Simple representation of a Home Assistant entity state."""

    entity_id: str
    state: str
    attributes: Mapping[str, Any]

    @property
    def domain_hint(self) -> str:
        return (self.entity_id.split(".", maxsplit=1)[0] if self.entity_id else "").lower()


class DeviceClassifier:
    """Classify entities into domains with alias and device_class support."""

    def __init__(self, aliases: Mapping[str, DeviceDomain] | None = None) -> None:
        self._aliases: dict[str, DeviceDomain] = {
            key.lower(): value for key, value in (aliases or _DEFAULT_ALIASES).items()
        }

    def classify(self, entity_id: str, attributes: Mapping[str, Any]) -> str:
        """Infer an entity's domain using entity_id, device_class, and aliases."""

        candidates = self._candidate_domains(entity_id, attributes)
        for candidate in candidates:
            normalized = self._normalize_domain(candidate)
            if normalized:
                return normalized

        # Fallback: scan aliases appearing in the entity_id for future domains.
        lowered_id = entity_id.lower() if entity_id else ""
        for alias, mapped_domain in self._aliases.items():
            if alias in lowered_id:
                return mapped_domain.value

        # Final fallback: return the raw domain hint or unknown to support new HA domains.
        domain_hint = candidates[-1] if candidates else ""
        return domain_hint or DeviceDomain.UNKNOWN.value

    def is_light(self, entity: EntityState) -> bool:
        return self._matches_domain(entity, DeviceDomain.LIGHT)

    def is_switch(self, entity: EntityState) -> bool:
        return self._matches_domain(entity, DeviceDomain.SWITCH)

    def is_lamp(self, entity: EntityState) -> bool:
        return self.is_light(entity)

    def is_cover(self, entity: EntityState) -> bool:
        return self._matches_domain(entity, DeviceDomain.COVER)

    def is_media_player(self, entity: EntityState) -> bool:
        return self._matches_domain(entity, DeviceDomain.MEDIA_PLAYER)

    def is_climate(self, entity: EntityState) -> bool:
        return self._matches_domain(entity, DeviceDomain.CLIMATE)

    def domain_supports_on_off(self, domain: str) -> bool:
        domain_enum = self._to_domain_enum(domain)
        return domain_enum in {
            DeviceDomain.LIGHT,
            DeviceDomain.SWITCH,
            DeviceDomain.FAN,
            DeviceDomain.HUMIDIFIER,
            DeviceDomain.MEDIA_PLAYER,
            DeviceDomain.COVER,
            DeviceDomain.VACUUM,
            DeviceDomain.LOCK,
        }

    def is_on_state(self, domain: str, state: str) -> bool:
        domain_enum = self._to_domain_enum(domain)
        normalized_state = state.lower()

        if domain_enum in {DeviceDomain.LIGHT, DeviceDomain.SWITCH, DeviceDomain.FAN, DeviceDomain.HUMIDIFIER}:
            return normalized_state == "on"
        if domain_enum == DeviceDomain.MEDIA_PLAYER:
            return normalized_state in {"on", "playing", "idle", "paused"}
        if domain_enum == DeviceDomain.COVER:
            return normalized_state in {"open", "opening"}
        if domain_enum == DeviceDomain.VACUUM:
            return normalized_state in {"on", "cleaning", "returning"}
        if domain_enum == DeviceDomain.LOCK:
            return normalized_state in {"unlocked", "open"}

        return False

    def _candidate_domains(
        self, entity_id: str, attributes: Mapping[str, Any]
    ) -> list[str]:
        candidates: list[str] = []
        device_class = str(attributes.get("device_class", "")).strip().lower()
        if device_class:
            candidates.append(device_class)

        if entity_id:
            domain_hint = entity_id.split(".", maxsplit=1)[0].lower()
            candidates.append(domain_hint)

        return candidates

    def _normalize_domain(self, candidate: str) -> str | None:
        candidate = candidate.lower()
        alias_match = self._aliases.get(candidate)
        if alias_match:
            return alias_match.value

        domain_enum = self._to_domain_enum(candidate)
        if domain_enum != DeviceDomain.UNKNOWN:
            return domain_enum.value
        return None

    def _matches_domain(self, entity: EntityState, domain: DeviceDomain) -> bool:
        return self._to_domain_enum(self.classify(entity.entity_id, entity.attributes)) == domain

    def _to_domain_enum(self, domain: str) -> DeviceDomain:
        try:
            return DeviceDomain(domain)
        except ValueError:
            return DeviceDomain.UNKNOWN


_DEFAULT_CLASSIFIER = DeviceClassifier()


def classify_entity(entity_id: str, attributes: Mapping[str, Any]) -> str:
    """Backward-compatible helper for existing code paths."""

    return _DEFAULT_CLASSIFIER.classify(entity_id, attributes)


def default_classifier() -> DeviceClassifier:
    """Expose the shared classifier instance."""

    return _DEFAULT_CLASSIFIER

