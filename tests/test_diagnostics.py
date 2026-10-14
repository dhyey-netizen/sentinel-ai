import pytest

from helpers import POOL_LOG
from sentinel_ai.diagnostics import LogCorpus, ToolError, parse_timestamp
from sentinel_ai.tools import ToolRegistry


def corpus(text: str = POOL_LOG) -> LogCorpus:
    return LogCorpus(text)


def test_parse_timestamp_variants():
    assert parse_timestamp("2026-03-14T10:15:01Z").isoformat() == "2026-03-14T10:15:01+00:00"
    assert parse_timestamp("2026-03-14 10:15:01,250").microsecond == 250000
    assert parse_timestamp("2026-03-14T10:15:01+05:30").hour == 4
    assert parse_timestamp("yesterday") is None
    assert parse_timestamp("2026-13-45T99:00:00Z") is None


def test_malformed_and_blank_lines_do_not_crash():
    c = LogCorpus("garbage\n\n   \n  at com.x.Y(Y.java:1)\n2026-03-14T10:00:00Z ERROR boom")
    assert c.stats.total_lines == 3
    assert c.stats.unparsed_lines == 2
    assert c.entries[-1].level == "ERROR"


def test_empty_corpus():
    c = LogCorpus("")
    assert c.stats.total_lines == 0
    assert c.summarize_errors().data["error_lines"] == 0


def test_long_line_truncation_is_flagged():
    c = LogCorpus("2026-03-14T10:00:00Z ERROR " + "x" * 500, max_line_chars=100)
    assert c.stats.truncated_lines == 1
    assert c.entries[0].truncated and len(c.entries[0].text) == 100


def test_search_requires_all_terms_and_reports_truncation():
    r = corpus().search_logs("connection timed", limit=2)
    assert r.data["total_matches"] == 3 and r.data["returned"] == 2 and r.truncated
    assert corpus().search_logs("nonexistent-term").entries == []
    with pytest.raises(ToolError):
        corpus().search_logs("   ")


def test_error_grouping_normalises_numbers_and_ids():
    r = corpus().summarize_errors()
    groups = r.data["groups"]
    assert len(groups) == 1
    assert groups[0]["count"] == 3
    assert r.data["level_counts"] == {"WARN": 1, "ERROR": 3, "INFO": 1}
    assert groups[0]["request_ids"] == ["req-1", "req-2", "req-3"]


def test_trace_request_exact_match_only():
    c = corpus()
    assert c.trace_request("req-1").data["total_lines"] == 1
    assert c.trace_request("req").data["total_lines"] == 0  # no prefix matching
    assert "no lines carry this request id" in c.trace_request("req-99").notes


def test_time_window_filters_and_validates():
    c = corpus()
    r = c.find_time_window("2026-03-14T10:15:28Z", "2026-03-14T10:15:30Z")
    assert r.data["total_lines"] == 3
    with pytest.raises(ToolError, match="after"):
        c.find_time_window("2026-03-14T11:00:00Z", "2026-03-14T10:00:00Z")
    with pytest.raises(ToolError, match="ISO-8601"):
        c.find_time_window("soon", "later-today")
    with pytest.raises(ToolError, match="no parseable timestamps"):
        LogCorpus("ERROR no timestamp here").find_time_window("2026-01-01", "2026-01-02")


def test_extract_error_context_returns_neighbours():
    r = corpus().extract_error_context("req-2", context_lines=1)
    assert [e.line_no for e in r.entries] == [2, 3, 4]
    assert "identifier not found in logs" in corpus().extract_error_context("zzz").notes


# ---- tool execution restrictions -------------------------------------------------


@pytest.mark.parametrize(
    "name,args",
    [
        ("shell", {"cmd": "ls"}),
        ("read_file", {"path": "/etc/passwd"}),
        ("search_logs", {"query": "x", "path": "/etc/passwd"}),  # extra arg rejected
        ("search_logs", {"limit": 5}),  # missing required arg
        ("search_logs", {"query": "x", "limit": 9999}),  # out of range
        ("trace_request", {"request_id": "req-1; DROP TABLE x"}),
        ("trace_request", {"request_id": "../../etc"}),
    ],
)
def test_registry_rejects_unsafe_or_invalid_calls(name, args):
    with pytest.raises(ToolError):
        ToolRegistry(corpus()).execute(name, args)


def test_registry_only_exposes_allowlisted_tools():
    from sentinel_ai.tools import TOOL_SPECS

    assert set(TOOL_SPECS) == {
        "search_logs",
        "summarize_errors",
        "trace_request",
        "find_time_window",
        "extract_error_context",
    }


def test_registry_scopes_to_its_own_corpus():
    a = ToolRegistry(LogCorpus("2026-03-14T10:00:00Z ERROR only-in-a"))
    b = ToolRegistry(LogCorpus("2026-03-14T10:00:00Z ERROR only-in-b"))
    assert a.execute("search_logs", {"query": "only-in-b"})[0].entries == []
    assert len(b.execute("search_logs", {"query": "only-in-b"})[0].entries) == 1
