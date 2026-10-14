"""Application service: validate -> redact -> investigate -> persist. Shared by CLI and API."""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import UTC, datetime

from sentinel_ai.agent import Investigator, initial_state, run_investigation
from sentinel_ai.config import Settings
from sentinel_ai.diagnostics import LogCorpus
from sentinel_ai.llm import LLMClient, OllamaLLM
from sentinel_ai.redaction import redact
from sentinel_ai.schemas import InvestigationRecord, InvestigationRequest, LogStats
from sentinel_ai.storage import InvestigationStore

log = logging.getLogger("sentinel_ai")


class InputError(ValueError):
    """The request is well-formed but violates a configured limit. Message is safe to show."""

    def __init__(self, message: str, *, too_large: bool = False) -> None:
        super().__init__(message)
        self.too_large = too_large


def validate_limits(request: InvestigationRequest, settings: Settings) -> None:
    """Reject (never silently truncate) inputs over the configured limits."""
    if len(request.question) > settings.max_question_chars:
        raise InputError(
            f"question is {len(request.question)} characters; the limit is "
            f"{settings.max_question_chars}",
            too_large=True,
        )
    size = len(request.log_text.encode("utf-8", errors="replace"))
    if size > settings.max_log_bytes:
        raise InputError(
            f"log input is {size} bytes; the limit is {settings.max_log_bytes} "
            "(set SENTINEL_MAX_LOG_BYTES to raise it, or supply a narrower time range)",
            too_large=True,
        )


class InvestigationService:
    def __init__(
        self, settings: Settings, store: InvestigationStore, llm: LLMClient | None = None
    ) -> None:
        self.settings = settings
        self.store = store
        self.llm: LLMClient = llm if llm is not None else OllamaLLM(settings)

    def investigate(self, request: InvestigationRequest) -> InvestigationRecord:
        validate_limits(request, self.settings)
        investigation_id = uuid.uuid4().hex[:12]
        created = datetime.now(UTC)
        started = time.perf_counter()

        clean_text, redaction_counts = redact(request.log_text)
        clean_question, _ = redact(request.question)
        raw_bytes = request.log_text.encode("utf-8", errors="replace")
        corpus = LogCorpus(clean_text, self.settings.max_line_chars)
        log_stats = LogStats(
            sha256=hashlib.sha256(raw_bytes).hexdigest(),
            bytes=len(raw_bytes),
            lines=corpus.stats.total_lines,
            redactions=redaction_counts,
        )

        investigator = Investigator(corpus, self.llm, self.settings)
        state = initial_state(clean_question, request.metadata)
        report = None
        error: str | None = None
        status = "failed"
        history = []
        try:
            final = run_investigation(investigator, state)
            report = final.get("report")
            history = final.get("tool_history", [])
            status = final.get("status", "failed")
            error = final.get("error")
            if status == "running":  # graph ended without a report: treat as failure
                status, error = "failed", error or "investigation ended without a report"
        except Exception as exc:  # noqa: BLE001 - system boundary: never leak internals
            log.exception("investigation %s crashed", investigation_id)
            status, error = "failed", f"internal error ({type(exc).__name__})"

        record = InvestigationRecord(
            id=investigation_id,
            created_at=created,
            question=clean_question,
            metadata=request.metadata,
            status=status,  # type: ignore[arg-type]
            duration_ms=int((time.perf_counter() - started) * 1000),
            report=report,
            tool_history=history,
            error=error,
            log_stats=log_stats,
        )
        self.store.save(record)
        # Audit line: ids, counts and status only. Never log content.
        log.info(
            "investigation id=%s status=%s tools=%d duration_ms=%d redactions=%s",
            record.id,
            record.status,
            len(history),
            record.duration_ms,
            redaction_counts,
        )
        return record
