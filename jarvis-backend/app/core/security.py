"""Security utilities for validating tool arguments before hitting Home Assistant."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

SanitizedValue = str | int | float | bool | list[Any] | dict[str, Any]


class SanitizationError(ValueError):
    """Raised when user-supplied tool arguments are unsafe."""


def _sanitize_value(value: Any) -> SanitizedValue:
    if isinstance(value, (str, int, float, bool)):
        return value.strip().replace("\n", " ").replace("\r", " ") if isinstance(value, str) else value

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        sanitized_list: list[SanitizedValue] = []
        for index, item in enumerate(value):
            if index >= 20:
                break
            sanitized_list.append(_sanitize_value(item))
        return sanitized_list

    if isinstance(value, Mapping):
        sanitized_dict: dict[str, SanitizedValue] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 20:
                break
            key_str = str(key)
            if key_str.startswith("__"):
                raise SanitizationError("Unsafe mapping key detected")
            sanitized_dict[key_str] = _sanitize_value(item)
        return sanitized_dict

    raise SanitizationError("Unsupported argument type provided")


def sanitize_tool_arguments(
    args: Mapping[str, Any] | None,
    allowed_keys: Iterable[str],
) -> dict[str, SanitizedValue]:
    """Return a filtered and sanitized argument dictionary."""

    allowed = {key for key in allowed_keys}
    if not allowed:
        raise SanitizationError("No allowed keys defined for the tool")

    if args is None:
        return {}

    sanitized: dict[str, SanitizedValue] = {}
    for key, value in args.items():
        if key not in allowed or key.startswith("__"):
            continue
        sanitized[key] = _sanitize_value(value)

    return sanitized
