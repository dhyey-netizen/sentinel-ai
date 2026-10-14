"""Defense-in-depth heuristics for untrusted log content and model output.

These are NOT the primary safety boundary. The primary boundary is structural: the agent has
no write-capable tools, tools only see the current log corpus, and all tool arguments are
validated. These heuristics add visibility (flagging) and strip obviously dangerous advice.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_INJECTION_PATTERNS = [
    re.compile(
        r"ignore\s+(?:all\s+|any\s+)?(?:the\s+)?(?:previous|prior|above)\s+instructions", re.I
    ),
    re.compile(r"disregard\s+(?:all\s+|the\s+)?(?:previous|prior|above|your)\s+instructions", re.I),
    re.compile(r"\b(?:system\s+prompt|developer\s+message)\b", re.I),
    re.compile(r"\byou\s+are\s+now\b", re.I),
    re.compile(r"\bnew\s+instructions?\s*:", re.I),
    re.compile(r"\bas\s+an?\s+ai\b", re.I),
    re.compile(r"(?:^|\s)(?:assistant|system)\s*:", re.I),
]

_UNSAFE_ACTION_PATTERNS = [
    re.compile(r"\brm\s+-[a-z]*[rf]", re.I),
    re.compile(r"\bmkfs\b|\bdd\s+if=", re.I),
    re.compile(r"\b(?:curl|wget)\b[^|\n]*\|\s*(?:sudo\s+)?(?:ba)?sh\b", re.I),
    re.compile(r"\bdrop\s+(?:table|database|schema)\b|\btruncate\s+table\b", re.I),
    re.compile(r"\bdelete\s+from\b", re.I),
    re.compile(r"\bchmod\s+-R\s+777\b", re.I),
    re.compile(r">\s*/dev/sd[a-z]", re.I),
    re.compile(r":\(\)\s*\{\s*:\|:&\s*\};:"),
]


def looks_like_injection(text: str) -> bool:
    """True if ``text`` resembles an attempt to instruct the model."""
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def count_injection_like(lines: Iterable[str]) -> int:
    return sum(1 for line in lines if looks_like_injection(line))


def is_unsafe_action(text: str) -> bool:
    """True if a suggested action contains a destructive shell/SQL pattern."""
    return any(p.search(text) for p in _UNSAFE_ACTION_PATTERNS)


def filter_actions(actions: Iterable[str]) -> tuple[list[str], int]:
    """Drop destructive suggestions. Returns ``(kept, removed_count)``."""
    kept: list[str] = []
    removed = 0
    for action in actions:
        if is_unsafe_action(action):
            removed += 1
        else:
            kept.append(action)
    return kept, removed
