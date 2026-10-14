"""Deterministic diagnostic tools over a single, immutable log corpus.

Nothing here touches the filesystem, network, shell, or database. The corpus is built from
text that has already been redacted.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

from sentinel_ai.schemas import LogEntry, ToolResult

ERROR_LEVELS = frozenset({"ERROR", "CRITICAL", "FATAL"})

_LINE_RE = re.compile(
    r"""^\s*(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)
        \s+(?P<level>DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|FATAL)\b""",
    re.I | re.X,
)
_REQ_KV_RE = re.compile(r"\b(?:request_id|req_id|trace_id|rid)=(?P<id>[\w.-]+)")
_REQ_BRACKET_RE = re.compile(r"\[(?P<id>req-[\w-]+)\]")
_TS_PARTS_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})[T ](?P<time>\d{2}:\d{2}:\d{2})(?:[.,](?P<frac>\d+))?"
    r"(?P<tz>Z|[+-]\d{2}:?\d{2})?$"
)

_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_HEX_RE = re.compile(r"\b0x[0-9a-f]+\b", re.I)
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_ID_RE = re.compile(r"\b(?:req|order|user|cust|txn|trace)[-_][\w-]+", re.I)
_WS_RE = re.compile(r"\s+")


class ToolError(ValueError):
    """Invalid tool name/arguments or an unanswerable query. Safe to show to the model."""


def parse_timestamp(value: str) -> datetime | None:
    """Parse an ISO-8601-like timestamp. Naive values are treated as UTC. None if unparseable."""
    m = _TS_PARTS_RE.match(value.strip())
    if not m:
        return None
    frac = (m.group("frac") or "0")[:6].ljust(6, "0")
    tz = m.group("tz") or "Z"
    if tz == "Z":
        tz = "+00:00"
    elif ":" not in tz:
        tz = f"{tz[:3]}:{tz[3:]}"
    try:
        return datetime.fromisoformat(f"{m.group('date')}T{m.group('time')}.{frac}{tz}").astimezone(
            UTC
        )
    except ValueError:
        return None


def parse_line(line_no: int, raw: str, max_chars: int) -> LogEntry:
    """Parse one physical line into a ``LogEntry`` (never raises)."""
    truncated = len(raw) > max_chars
    text = raw[:max_chars] if truncated else raw
    header = _LINE_RE.match(text)
    if not header:
        return LogEntry(line_no=line_no, text=text, truncated=truncated)
    level = header.group("level").upper()
    if level == "WARNING":
        level = "WARN"
    req = _REQ_KV_RE.search(text) or _REQ_BRACKET_RE.search(text)
    return LogEntry(
        line_no=line_no,
        timestamp=parse_timestamp(header.group("ts")),
        level=level,
        request_id=req.group("id") if req else None,
        text=text,
        truncated=truncated,
    )


def error_signature(entry: LogEntry) -> str:
    """Normalise an error line so that recurring errors group together."""
    body = _LINE_RE.sub("", entry.text, count=1)
    body = _REQ_KV_RE.sub("", body)
    body = _REQ_BRACKET_RE.sub("", body)
    body = _UUID_RE.sub("<uuid>", body)
    body = _HEX_RE.sub("<hex>", body)
    body = _ID_RE.sub("<id>", body)
    body = _NUM_RE.sub("<n>", body)
    return _WS_RE.sub(" ", body).strip()[:160]


def _iso(ts: datetime | None) -> str | None:
    return ts.isoformat().replace("+00:00", "Z") if ts else None


@dataclass(frozen=True)
class CorpusStats:
    total_lines: int
    leveled_entries: int
    unparsed_lines: int
    truncated_lines: int
    has_timestamps: bool


class LogCorpus:
    """Parsed log lines for ONE investigation, with read-only query methods (the tools)."""

    def __init__(self, text: str, max_line_chars: int = 2000) -> None:
        entries: list[LogEntry] = []
        for line_no, raw in enumerate(text.splitlines(), start=1):
            if raw.strip():
                entries.append(parse_line(line_no, raw.rstrip(), max_line_chars))
        self._entries: tuple[LogEntry, ...] = tuple(entries)
        self._by_line: dict[int, LogEntry] = {e.line_no: e for e in entries}
        leveled = sum(1 for e in entries if e.level)
        self.stats = CorpusStats(
            total_lines=len(entries),
            leveled_entries=leveled,
            unparsed_lines=len(entries) - leveled,
            truncated_lines=sum(1 for e in entries if e.truncated),
            has_timestamps=any(e.timestamp for e in entries),
        )

    @property
    def entries(self) -> tuple[LogEntry, ...]:
        return self._entries

    def get_line(self, line_no: int) -> LogEntry | None:
        return self._by_line.get(line_no)

    # ------------------------------------------------------------------ tools

    def search_logs(self, query: str, limit: int = 20) -> ToolResult:
        """Case-insensitive search. Every whitespace-separated term must appear in the line."""
        terms = query.lower().split()
        if not terms:
            raise ToolError("query must not be empty")
        limit = max(1, min(limit, 50))
        matches = [e for e in self._entries if all(t in e.text.lower() for t in terms)]
        hits = matches[:limit]
        return ToolResult(
            tool="search_logs",
            data={"query": query, "total_matches": len(matches), "returned": len(hits)},
            entries=hits,
            truncated=len(matches) > len(hits),
            notes=[f"{len(matches) - len(hits)} further matches not returned; narrow the query"]
            if len(matches) > len(hits)
            else [],
        )

    def summarize_errors(self, top_n: int = 8) -> ToolResult:
        """Group ERROR/CRITICAL/FATAL lines by normalised signature; add level counts and spikes."""
        level_counts: Counter[str] = Counter(e.level for e in self._entries if e.level)
        groups: dict[str, list[LogEntry]] = defaultdict(list)
        for e in self._entries:
            if e.level in ERROR_LEVELS:
                groups[error_signature(e)].append(e)
        ranked = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[1][0].line_no))
        group_rows = []
        examples: list[LogEntry] = []
        for sig, members in ranked[:top_n]:
            stamps = [m.timestamp for m in members if m.timestamp]
            group_rows.append(
                {
                    "signature": sig,
                    "count": len(members),
                    "first_seen": _iso(min(stamps)) if stamps else None,
                    "last_seen": _iso(max(stamps)) if stamps else None,
                    "example_line": members[0].line_no,
                    "request_ids": sorted({m.request_id for m in members if m.request_id})[:3],
                }
            )
            examples.append(members[0])

        per_minute_err: Counter[str] = Counter()
        per_minute_all: Counter[str] = Counter()
        for e in self._entries:
            if e.timestamp:
                bucket = e.timestamp.strftime("%Y-%m-%dT%H:%MZ")
                per_minute_all[bucket] += 1
                if e.level in ERROR_LEVELS:
                    per_minute_err[bucket] += 1
        spikes = [
            {"minute": m, "errors": n, "total_lines": per_minute_all[m]}
            for m, n in sorted(per_minute_err.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        ]
        error_total = sum(len(v) for v in groups.values())
        return ToolResult(
            tool="summarize_errors",
            data={
                "level_counts": dict(level_counts),
                "error_lines": error_total,
                "distinct_error_signatures": len(groups),
                "groups": group_rows,
                "busiest_error_minutes": spikes,
            },
            entries=examples,
            truncated=len(ranked) > top_n,
            notes=[f"{len(ranked) - top_n} less frequent error signatures omitted"]
            if len(ranked) > top_n
            else [],
        )

    def trace_request(self, request_id: str, limit: int = 50) -> ToolResult:
        """All lines carrying ``request_id`` (exact id match), in file order."""
        limit = max(1, min(limit, 100))
        matches = [e for e in self._entries if e.request_id == request_id]
        hits = matches[:limit]
        stamps = [e.timestamp for e in matches if e.timestamp]
        return ToolResult(
            tool="trace_request",
            data={
                "request_id": request_id,
                "total_lines": len(matches),
                "first_seen": _iso(min(stamps)) if stamps else None,
                "last_seen": _iso(max(stamps)) if stamps else None,
                "levels": dict(Counter(e.level for e in matches if e.level)),
            },
            entries=hits,
            truncated=len(matches) > len(hits),
            notes=[] if matches else ["no lines carry this request id"],
        )

    def find_time_window(self, start_time: str, end_time: str, limit: int = 30) -> ToolResult:
        """Lines whose parsed timestamp lies in [start_time, end_time]; WARN/ERROR lines first."""
        if not self.stats.has_timestamps:
            raise ToolError("log corpus has no parseable timestamps")
        start, end = parse_timestamp(start_time), parse_timestamp(end_time)
        if start is None or end is None:
            raise ToolError("start_time/end_time must be ISO-8601, e.g. 2026-03-14T10:15:00Z")
        if start > end:
            raise ToolError("start_time must not be after end_time")
        limit = max(1, min(limit, 50))
        inside = [e for e in self._entries if e.timestamp and start <= e.timestamp <= end]
        severity = {"CRITICAL": 0, "FATAL": 0, "ERROR": 1, "WARN": 2}
        ordered = sorted(inside, key=lambda e: (severity.get(e.level or "", 3), e.line_no))
        hits = sorted(ordered[:limit], key=lambda e: e.line_no)
        return ToolResult(
            tool="find_time_window",
            data={
                "start_time": start_time,
                "end_time": end_time,
                "total_lines": len(inside),
                "returned": len(hits),
                "levels": dict(Counter(e.level for e in inside if e.level)),
            },
            entries=hits,
            truncated=len(inside) > len(hits),
            notes=["most severe lines were prioritised"] if len(inside) > len(hits) else [],
        )

    def extract_error_context(
        self, error_identifier: str, context_lines: int = 3, max_occurrences: int = 3
    ) -> ToolResult:
        """Surrounding lines for the first occurrences of an error string or exception class."""
        needle = error_identifier.lower().strip()
        if not needle:
            raise ToolError("error_identifier must not be empty")
        context_lines = max(0, min(context_lines, 10))
        positions = [i for i, e in enumerate(self._entries) if needle in e.text.lower()]
        chosen = positions[:max_occurrences]
        seen: dict[int, LogEntry] = {}
        for pos in chosen:
            lo, hi = max(0, pos - context_lines), min(len(self._entries), pos + context_lines + 1)
            for e in self._entries[lo:hi]:
                seen[e.line_no] = e
        return ToolResult(
            tool="extract_error_context",
            data={
                "error_identifier": error_identifier,
                "total_occurrences": len(positions),
                "occurrences_shown": len(chosen),
            },
            entries=[seen[k] for k in sorted(seen)],
            truncated=len(positions) > len(chosen),
            notes=[] if positions else ["identifier not found in logs"],
        )
