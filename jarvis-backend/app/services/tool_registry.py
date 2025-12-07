"""Canonical definitions for Home Assistant tool calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from app.core.security import SanitizationError, sanitize_tool_arguments


@dataclass(slots=True, frozen=True)
class ToolDefinition:
    name: str
    description: str
    domain: str
    service: str
    allowed_args: tuple[str, ...]
    required_args: tuple[str, ...] = ("entity_id",)


TOOL_DEFINITIONS: dict[str, ToolDefinition] = {
    definition.name: definition
    for definition in (
        ToolDefinition(
            name="ha_light_turn_on",
            description="Turn on or adjust a light entity.",
            domain="light",
            service="turn_on",
            allowed_args=("entity_id", "brightness", "brightness_pct", "color_temp"),
        ),
        ToolDefinition(
            name="ha_light_turn_off",
            description="Turn off a light entity.",
            domain="light",
            service="turn_off",
            allowed_args=("entity_id",),
        ),
        ToolDefinition(
            name="ha_switch_turn_on",
            description="Turn on a switch or smart plug.",
            domain="switch",
            service="turn_on",
            allowed_args=("entity_id",),
        ),
        ToolDefinition(
            name="ha_switch_turn_off",
            description="Turn off a switch or smart plug.",
            domain="switch",
            service="turn_off",
            allowed_args=("entity_id",),
        ),
        ToolDefinition(
            name="ha_media_play",
            description="Resume or start media playback.",
            domain="media_player",
            service="media_play",
            allowed_args=("entity_id",),
        ),
        ToolDefinition(
            name="ha_media_pause",
            description="Pause or stop media playback.",
            domain="media_player",
            service="media_pause",
            allowed_args=("entity_id",),
        ),
        ToolDefinition(
            name="ha_cover_open",
            description="Open a cover, blind, or shade.",
            domain="cover",
            service="open_cover",
            allowed_args=("entity_id",),
        ),
        ToolDefinition(
            name="ha_cover_close",
            description="Close a cover, blind, or shade.",
            domain="cover",
            service="close_cover",
            allowed_args=("entity_id",),
        ),
        ToolDefinition(
            name="ha_climate_set_temperature",
            description="Set target temperature for a climate device.",
            domain="climate",
            service="set_temperature",
            allowed_args=("entity_id", "temperature"),
            required_args=("entity_id", "temperature"),
        ),
    )
}

_INTENT_TOOL_MAPPING: dict[str, tuple[str, str]] = {
    "light_control": ("ha_light_turn_on", "ha_light_turn_off"),
    "switch_control": ("ha_switch_turn_on", "ha_switch_turn_off"),
    "media_control": ("ha_media_play", "ha_media_pause"),
    "cover_control": ("ha_cover_open", "ha_cover_close"),
    "climate_control": ("ha_climate_set_temperature", "ha_climate_set_temperature"),
}


def get_tool_definition(name: str) -> ToolDefinition | None:
    return TOOL_DEFINITIONS.get(name)


def tool_for_intent(intent: str, *, turn_off: bool = False) -> ToolDefinition | None:
    tools = _INTENT_TOOL_MAPPING.get(intent)
    if not tools:
        return None
    tool_name = tools[1] if turn_off else tools[0]
    return get_tool_definition(tool_name)


def describe_tools_for_prompt() -> str:
    """Return Markdown description of tools for LLM conditioning."""

    lines: list[str] = []
    for definition in TOOL_DEFINITIONS.values():
        allowed = ", ".join(definition.allowed_args)
        lines.append(
            f"- {definition.name}: {definition.description} (allowed args: {allowed})"
        )
    return "\n".join(lines)


def prepare_service_payload(
    tool_name: str,
    args: Mapping[str, Any] | None,
) -> tuple[str, str, dict[str, Any]]:
    """Validate tool arguments and map to Home Assistant service metadata."""

    definition = get_tool_definition(tool_name)
    if not definition:
        raise ValueError(f"Unknown tool: {tool_name}")

    sanitized_args = sanitize_tool_arguments(args or {}, definition.allowed_args)
    for key in definition.required_args:
        if key not in sanitized_args:
            raise SanitizationError(f"Missing required argument '{key}' for {tool_name}")

    return definition.domain, definition.service, sanitized_args
