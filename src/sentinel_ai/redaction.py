"""Secret redaction applied to log text BEFORE it reaches the model, tools, or storage.

Pattern-based redaction is best-effort: it catches common secret shapes, not every secret.
"""

from __future__ import annotations

import re

_REDACTION_RULES: list[tuple[str, re.Pattern[str], str]] = [
    (
        "private_key",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*", re.I),
        "[REDACTED:private_key]",
    ),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED:aws_key]"),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "[REDACTED:github_token]"),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"), "[REDACTED:slack_token]"),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b"),
        "[REDACTED:jwt]",
    ),
    (
        "bearer_token",
        re.compile(r"\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{8,}", re.I),
        r"\1 [REDACTED:token]",
    ),
    (
        "url_credentials",
        re.compile(r"(\b[a-z][a-z0-9+.-]*://)[^\s/@:]+:[^\s/@]+@", re.I),
        r"\1[REDACTED:credentials]@",
    ),
    (
        "key_value_secret",
        re.compile(
            r"""(?ix)
            \b(password|passwd|pwd|secret|client_secret|token|access_token|refresh_token|
               api[_-]?key|apikey|access[_-]?key|private[_-]?key|authorization|cookie|session[_-]?id)
            (\s*[=:]\s*)
            ("[^"]*"|'[^']*'|[^\s,;&"']+)
            """
        ),
        r"\1\2[REDACTED:secret]",
    ),
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED:email]",
    ),
]


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Return ``(redacted_text, counts_by_rule)``. Counts never include the secret values."""
    counts: dict[str, int] = {}
    for name, pattern, replacement in _REDACTION_RULES:
        text, n = pattern.subn(replacement, text)
        if n:
            counts[name] = counts.get(name, 0) + n
    return text, counts
