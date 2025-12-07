"""Keyword-based semantic routing for fast-path commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class SemanticRoute:
    intent: str
    domain: str
    requires_llm: bool = False
    keywords: tuple[str, ...] = ()


_DIRECT_PATTERNS: dict[str, tuple[str, ...]] = {
    "light_control": ("light", "lights", "licht", "lampe", "lampen"),
    "cover_control": ("rollo", "rolladen", "shade", "blinds", "jalousie"),
    "media_control": ("musik", "speaker", "musik", "tv", "fernseher", "media"),
    "climate_control": ("klima", "thermostat", "heizung", "heizer", "ac", "ventilator"),
    "switch_control": ("steckdose", "switch", "power", "schalte"),
}

_INTENT_DOMAIN = {
    "light_control": "light",
    "cover_control": "cover",
    "media_control": "media_player",
    "climate_control": "climate",
    "switch_control": "switch",
}


def _contains_keyword(text: str, keywords: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def semantic_route(text: str) -> SemanticRoute | None:
    """Return a semantic route for simple commands if we can."""

    for intent, keywords in _DIRECT_PATTERNS.items():
        if _contains_keyword(text, keywords):
            domain = _INTENT_DOMAIN[intent]
            return SemanticRoute(
                intent=intent,
                domain=domain,
                requires_llm=False,
                keywords=keywords,
            )
    return None
