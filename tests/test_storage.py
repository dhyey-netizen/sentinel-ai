from datetime import UTC, datetime

from sentinel_ai.schemas import InvestigationRecord, LogStats, ToolExecution
from sentinel_ai.storage import InvestigationStore


def make(id_: str, minute: int) -> InvestigationRecord:
    return InvestigationRecord(
        id=id_,
        created_at=datetime(2026, 3, 14, 10, minute, tzinfo=UTC),
        question=f"question {id_}",
        metadata={"service": "checkout"},
        status="completed",
        duration_ms=123,
        tool_history=[
            ToolExecution(step=1, tool="summarize_errors", arguments={}, ok=True, summary="ok")
        ],
        log_stats=LogStats(sha256="ab" * 32, bytes=10, lines=2, redactions={"jwt": 1}),
    )


def test_roundtrip_and_missing(tmp_path):
    store = InvestigationStore(tmp_path / "nested" / "s.db")
    store.save(make("a1", 1))
    got = store.get("a1")
    assert got is not None and got.metadata == {"service": "checkout"}
    assert got.tool_history[0].tool == "summarize_errors"
    assert got.log_stats.redactions == {"jwt": 1}
    assert store.get("missing") is None


def test_list_newest_first_and_limit(tmp_path):
    store = InvestigationStore(tmp_path / "s.db")
    for i, minute in enumerate([5, 30, 10]):
        store.save(make(f"id{i}", minute))
    assert [s.id for s in store.list()] == ["id1", "id2", "id0"]
    assert len(store.list(limit=2)) == 2


def test_persists_across_instances(tmp_path):
    InvestigationStore(tmp_path / "s.db").save(make("x", 1))
    assert InvestigationStore(tmp_path / "s.db").get("x") is not None


def test_sql_injection_in_id_is_inert(tmp_path):
    store = InvestigationStore(tmp_path / "s.db")
    store.save(make("a1", 1))
    assert store.get("a1' OR '1'='1") is None
    assert len(store.list()) == 1
