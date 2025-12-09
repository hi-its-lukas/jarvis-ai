"""Function registry describing Home Assistant compatible function calls."""

from __future__ import annotations

FUNCTION_DEFINITIONS = [
    {
        "name": "turn_on",
        "parameters": {
            "type": "object",
            "properties": {"entity_id": {"type": "string"}},
            "required": ["entity_id"],
        },
    },
    {
        "name": "turn_off",
        "parameters": {
            "type": "object",
            "properties": {"entity_id": {"type": "string"}},
            "required": ["entity_id"],
        },
    },
    {
        "name": "open_cover",
        "parameters": {
            "type": "object",
            "properties": {"entity_id": {"type": "string"}},
            "required": ["entity_id"],
        },
    },
    {
        "name": "close_cover",
        "parameters": {
            "type": "object",
            "properties": {"entity_id": {"type": "string"}},
            "required": ["entity_id"],
        },
    },
    {
        "name": "set_brightness",
        "parameters": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string"},
                "brightness": {"type": "integer"},
            },
            "required": ["entity_id"],
        },
    },
    {
        "name": "set_temperature",
        "parameters": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string"},
                "temperature": {"type": "number"},
            },
            "required": ["entity_id", "temperature"],
        },
    },
    {
        "name": "media_play_pause",
        "parameters": {
            "type": "object",
            "properties": {"entity_id": {"type": "string"}},
            "required": ["entity_id"],
        },
    },
    {
        "name": "set_volume",
        "parameters": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string"},
                "volume": {"type": "number"},
            },
            "required": ["entity_id", "volume"],
        },
    },
    {
        "name": "set_color",
        "parameters": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string"},
                "rgb_color": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
            },
            "required": ["entity_id", "rgb_color"],
        },
    },
]

