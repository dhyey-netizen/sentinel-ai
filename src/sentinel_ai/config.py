"""Runtime configuration, read from environment variables (no secrets, no cloud keys)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    """Raised when an environment variable has an invalid value."""


def _get(env: Mapping[str, str], name: str, default: str) -> str:
    return env.get(name, default).strip()


def _get_int(env: Mapping[str, str], name: str, default: int, minimum: int = 1) -> int:
    raw = _get(env, name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def _get_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = _get(env, name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """All tunables. Defaults are chosen for qwen3:4b on a 6 GB GPU."""

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    ollama_num_ctx: int = 8192  # KV cache for 8k ctx fits alongside the ~2.5 GB Q4 weights
    ollama_temperature: float = 0.0
    ollama_timeout_s: float = 300.0
    db_path: Path = Path("data/sentinel.db")
    max_log_bytes: int = 2_000_000
    max_question_chars: int = 1_000
    max_line_chars: int = 2_000
    max_iterations: int = 4
    max_evidence: int = 60
    llm_evidence_chars: int = 9_000
    report_max_evidence: int = 20

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from ``SENTINEL_*`` environment variables."""
        e = os.environ if env is None else env
        return cls(
            ollama_base_url=_get(e, "SENTINEL_OLLAMA_URL", cls.ollama_base_url),
            ollama_model=_get(e, "SENTINEL_OLLAMA_MODEL", cls.ollama_model),
            ollama_num_ctx=_get_int(e, "SENTINEL_OLLAMA_NUM_CTX", cls.ollama_num_ctx, 1024),
            ollama_temperature=_get_float(e, "SENTINEL_OLLAMA_TEMPERATURE", cls.ollama_temperature),
            ollama_timeout_s=_get_float(e, "SENTINEL_OLLAMA_TIMEOUT_S", cls.ollama_timeout_s),
            db_path=Path(_get(e, "SENTINEL_DB_PATH", str(cls.db_path))),
            max_log_bytes=_get_int(e, "SENTINEL_MAX_LOG_BYTES", cls.max_log_bytes),
            max_question_chars=_get_int(e, "SENTINEL_MAX_QUESTION_CHARS", cls.max_question_chars),
            max_line_chars=_get_int(e, "SENTINEL_MAX_LINE_CHARS", cls.max_line_chars, 80),
            max_iterations=_get_int(e, "SENTINEL_MAX_ITERATIONS", cls.max_iterations),
            max_evidence=_get_int(e, "SENTINEL_MAX_EVIDENCE", cls.max_evidence),
            llm_evidence_chars=_get_int(
                e, "SENTINEL_LLM_EVIDENCE_CHARS", cls.llm_evidence_chars, 500
            ),
            report_max_evidence=_get_int(
                e, "SENTINEL_REPORT_MAX_EVIDENCE", cls.report_max_evidence
            ),
        )
