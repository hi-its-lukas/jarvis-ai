"""Entity classification helpers for Home Assistant devices."""

from __future__ import annotations


def classify_entity(entity_id: str, attributes: dict) -> str:
    """
    Classify a Home Assistant entity into a simplified device class.

    Parameters
    ----------
    entity_id: str
        The full Home Assistant entity id (e.g. ``"light.kitchen"``).
    attributes: dict
        Entity attributes (unused for now but kept for future expansion).

    Returns
    -------
    str
        One of: "light", "switch", "cover", "media_player", "climate",
        "fan", "binary_sensor", "sensor", or "unknown".
    """

    domain = (entity_id or "").split(".")[0]

    mapping = {
        "light": "light",
        "switch": "switch",
        "cover": "cover",
        "media_player": "media_player",
        "fan": "fan",
        "climate": "climate",
        "sensor": "sensor",
        "binary_sensor": "binary_sensor",
    }

    return mapping.get(domain, "unknown")

