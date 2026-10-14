"""SQLite persistence for investigations.

Raw logs are NOT stored: only a SHA-256 digest, size and redaction counts (see LogStats).
Report evidence excerpts are already redacted.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from sentinel_ai.schemas import InvestigationRecord, InvestigationSummary

_SCHEMA = """
CREATE TABLE IF NOT EXISTS investigations (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    question    TEXT NOT NULL,
    status      TEXT NOT NULL,
    duration_ms INTEGER NOT NULL,
    record_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_investigations_created ON investigations(created_at DESC);
"""


class InvestigationStore:
    """Small repository around one SQLite file. A new connection is opened per operation."""

    def __init__(self, db_path: Path | str) -> None:
        self._path = str(db_path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._memory_conn: sqlite3.Connection | None = None
        if self._path == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if self._memory_conn is not None:
            with self._memory_conn as conn:
                yield conn
            return
        with closing(sqlite3.connect(self._path, timeout=10)) as conn, conn:
            yield conn

    def save(self, record: InvestigationRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO investigations "
                "(id, created_at, question, status, duration_ms, record_json) VALUES (?,?,?,?,?,?)",
                (
                    record.id,
                    record.created_at.isoformat(),
                    record.question,
                    record.status,
                    record.duration_ms,
                    record.model_dump_json(),
                ),
            )

    def get(self, investigation_id: str) -> InvestigationRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT record_json FROM investigations WHERE id = ?", (investigation_id,)
            ).fetchone()
        return InvestigationRecord.model_validate_json(row[0]) if row else None

    def list(self, limit: int = 20) -> list[InvestigationSummary]:
        limit = max(1, min(limit, 200))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, created_at, question, status, duration_ms FROM investigations "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            InvestigationSummary(
                id=r[0], created_at=r[1], question=r[2], status=r[3], duration_ms=r[4]
            )
            for r in rows
        ]
