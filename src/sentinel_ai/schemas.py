"""Pydantic models shared by the agent, storage, API and CLI."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Confidence = Literal["insufficient", "low", "medium", "high"]
CONFIDENCE_RANK: dict[str, int] = {"insufficient": 0, "low": 1, "medium": 2, "high": 3}
Causation = Literal["supported_causal", "correlation_only"]
Status = Literal["completed", "completed_degraded", "failed"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- input


class InvestigationRequest(StrictModel):
    """Validated user input. Size limits beyond basic shape are enforced in ``service``."""

    question: str = Field(min_length=3, description="The incident question to investigate.")
    log_text: str = Field(description="Raw application log text.")
    metadata: dict[str, str] = Field(default_factory=dict, description="Optional incident tags.")

    @field_validator("question")
    @classmethod
    def _strip_question(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("question must contain at least 3 non-space characters")
        return v

    @field_validator("log_text")
    @classmethod
    def _non_empty_logs(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("log_text is empty")
        return v

    @field_validator("metadata")
    @classmethod
    def _bounded_metadata(cls, v: dict[str, str]) -> dict[str, str]:
        if len(v) > 20:
            raise ValueError("metadata may contain at most 20 keys")
        for key, value in v.items():
            if len(key) > 64 or len(value) > 256:
                raise ValueError("metadata keys must be <= 64 chars and values <= 256 chars")
        return v


# --------------------------------------------------------------------------- log data


class LogEntry(StrictModel):
    """One physical log line. Lines that do not match the header format have ``level=None``."""

    line_no: int
    timestamp: datetime | None = None
    level: str | None = None
    request_id: str | None = None
    text: str
    truncated: bool = False


class ToolResult(StrictModel):
    """Uniform result of a diagnostic tool."""

    tool: str
    data: dict[str, Any] = Field(default_factory=dict)
    entries: list[LogEntry] = Field(default_factory=list)
    truncated: bool = False
    notes: list[str] = Field(default_factory=list)


class ToolExecution(StrictModel):
    """Compact audit record of one tool call (arguments only, never raw logs)."""

    step: int
    tool: str
    arguments: dict[str, Any]
    ok: bool
    summary: str
    duration_ms: int = 0
    error: str | None = None


# --------------------------------------------------------------------------- evidence


class Evidence(StrictModel):
    id: str
    line_no: int
    timestamp: str | None = None
    level: str | None = None
    text: str
    found_by: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- LLM output


class ToolChoice(BaseModel):
    """LLM decision for the next step. Validated further against the tool registry."""

    model_config = ConfigDict(extra="ignore")

    action: Literal["call_tool", "finish"]
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class HypothesisDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field(max_length=200)
    explanation: str = Field(max_length=1500)
    evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    confidence: Confidence = "low"
    causation: Causation = "correlation_only"


class AnalysisDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    incident_summary: str = Field(max_length=1500)
    observed_symptoms: list[str] = Field(default_factory=list, max_length=10)
    hypotheses: list[HypothesisDraft] = Field(default_factory=list, max_length=5)
    recommended_next_steps: list[str] = Field(default_factory=list, max_length=10)
    suggested_remediation: list[str] = Field(default_factory=list, max_length=10)
    unknowns: list[str] = Field(default_factory=list, max_length=10)


# --------------------------------------------------------------------------- report


class RankedHypothesis(StrictModel):
    title: str
    explanation: str
    evidence_ids: list[str]
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    model_confidence: Confidence
    confidence: Confidence
    causation: Causation
    review_notes: list[str] = Field(default_factory=list)


class Report(StrictModel):
    """Final structured incident report. Evidence excerpts come from the log corpus, not the LLM."""

    incident_summary: str
    observed_symptoms: list[str]
    evidence: list[Evidence]
    root_cause: RankedHypothesis | None
    alternative_hypotheses: list[RankedHypothesis]
    confidence: Confidence
    confidence_explanation: str
    recommended_next_steps: list[str]
    suggested_remediation: list[str] = Field(
        description="UNVERIFIED suggestions. Never executed automatically; require human approval."
    )
    unknowns_and_limitations: list[str]
    insufficient_evidence: bool
    remediation_requires_human_approval: Literal[True] = True


# --------------------------------------------------------------------------- persistence


class LogStats(StrictModel):
    sha256: str
    bytes: int
    lines: int
    redactions: dict[str, int] = Field(default_factory=dict)


class InvestigationRecord(StrictModel):
    id: str
    created_at: datetime
    question: str
    metadata: dict[str, str] = Field(default_factory=dict)
    status: Status
    duration_ms: int
    report: Report | None = None
    tool_history: list[ToolExecution] = Field(default_factory=list)
    error: str | None = None
    log_stats: LogStats | None = None


class InvestigationSummary(StrictModel):
    id: str
    created_at: datetime
    question: str
    status: Status
    duration_ms: int
