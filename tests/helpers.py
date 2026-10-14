"""Shared test doubles. No network, no real model."""

from __future__ import annotations

from typing import Any

from sentinel_ai.config import Settings
from sentinel_ai.llm import LLMError

POOL_LOG = """\
2026-03-14T10:15:01.000Z WARN  db-pool pool at 100% (active=20, waiting=23)
2026-03-14T10:15:28.000Z ERROR checkout request_id=req-1 Connection unavailable, timed out (wait=41)
2026-03-14T10:15:29.000Z ERROR checkout request_id=req-2 Connection unavailable, timed out (wait=44)
2026-03-14T10:15:30.000Z ERROR checkout request_id=req-3 Connection unavailable, timed out (wait=47)
2026-03-14T10:15:31.000Z INFO  search request_id=req-4 GET /search ok
"""


class ScriptedLLM:
    """Replays plan responses in order; answers analysis prompts with ``analysis``.

    A plan response may be a dict, or an Exception instance to raise. When the script runs out
    the planner answers ``finish``. ``analysis`` may be a dict or an Exception.
    """

    def __init__(self, plans: list[Any] | None = None, analysis: Any = None) -> None:
        self.plans = list(plans or [])
        self.analysis = analysis
        self.prompts: list[str] = []
        self.plan_calls = 0

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        self.prompts.append(user)
        if "Available tools" in user:
            self.plan_calls += 1
            item = self.plans.pop(0) if self.plans else {"action": "finish"}
        else:
            item = self.analysis
            if item is None:
                raise LLMError("Ollama call failed: no analysis scripted")
        if isinstance(item, Exception):
            raise item
        return item


def good_analysis(ids: list[str]) -> dict[str, Any]:
    return {
        "incident_summary": "Checkout fails because DB connections are unavailable.",
        "observed_symptoms": ["Connection is not available errors"],
        "hypotheses": [
            {
                "title": "Connection pool exhaustion",
                "explanation": "Errors report no connection available while the pool is at 100%.",
                "evidence_ids": ids,
                "confidence": "high",
                "causation": "supported_causal",
            }
        ],
        "recommended_next_steps": ["Check which code path holds connections the longest."],
        "suggested_remediation": ["Consider raising the pool size after review."],
        "unknowns": ["Why connections are held so long is not visible in these logs."],
    }


def settings(**overrides: Any) -> Settings:
    return Settings(**{"max_iterations": 3, **overrides})
