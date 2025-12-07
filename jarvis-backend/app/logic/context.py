"""Simple in-memory context storage for recent interactions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable


@dataclass(slots=True)
class ContextEntry:
    user_text: str
    action_summary: str


class ConversationContext:
    def __init__(self, max_length: int = 10) -> None:
        self._history: Deque[ContextEntry] = deque(maxlen=max_length)

    def add_entry(self, user_text: str, action_summary: str) -> None:
        self._history.append(ContextEntry(user_text=user_text, action_summary=action_summary))

    def as_prompt_block(self) -> str:
        if not self._history:
            return ""
        lines = ["Recent context:"]
        for entry in self._history:
            lines.append(f"User: {entry.user_text}")
            lines.append(f"Action: {entry.action_summary}")
        return "\n".join(lines)

    def __iter__(self) -> Iterable[ContextEntry]:
        return iter(self._history)
