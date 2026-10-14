import pytest
from pydantic import ValidationError

from helpers import POOL_LOG, ScriptedLLM, good_analysis, settings
from sentinel_ai.schemas import InvestigationRequest
from sentinel_ai.service import InputError, InvestigationService
from sentinel_ai.storage import InvestigationStore

pytest.importorskip("langgraph")


def service(tmp_path, llm=None, **cfg):
    return InvestigationService(
        settings(**cfg),
        InvestigationStore(tmp_path / "t.db"),
        llm or ScriptedLLM(analysis=good_analysis(["E2", "E3"])),
    )


def req(log=POOL_LOG, question="Why are checkouts failing?", **kw):
    return InvestigationRequest(question=question, log_text=log, **kw)


@pytest.mark.parametrize("question,log", [("", "x"), ("  ", "x"), ("why?", ""), ("why?", "   \n ")])
def test_empty_inputs_rejected_by_schema(question, log):
    with pytest.raises(ValidationError):
        InvestigationRequest(question=question, log_text=log)


def test_unknown_fields_and_bad_metadata_rejected():
    with pytest.raises(ValidationError):
        InvestigationRequest(question="why?", log_text="x", shell="ls")
    with pytest.raises(ValidationError):
        InvestigationRequest(question="why?", log_text="x", metadata={"k": "v" * 300})


def test_oversized_log_is_rejected_not_truncated(tmp_path):
    with pytest.raises(InputError, match="limit") as exc:
        service(tmp_path, max_log_bytes=50).investigate(req())
    assert exc.value.too_large


def test_oversized_question_rejected(tmp_path):
    with pytest.raises(InputError):
        service(tmp_path, max_question_chars=10).investigate(req(question="q" * 50))


def test_secrets_never_reach_model_or_database(tmp_path):
    secret_log = POOL_LOG + "2026-03-14T10:15:40Z ERROR checkout password=hunter2-UNIQUE failed\n"
    llm = ScriptedLLM(analysis=good_analysis(["E2", "E3"]))
    svc = service(tmp_path, llm)
    record = svc.investigate(req(log=secret_log, question="why? token=abcSECRETtoken99"))
    assert not any("hunter2-UNIQUE" in p or "abcSECRETtoken99" in p for p in llm.prompts)
    stored = svc.store.get(record.id).model_dump_json()
    assert "hunter2-UNIQUE" not in stored and "abcSECRETtoken99" not in stored
    assert record.log_stats.redactions["key_value_secret"] == 1  # question is redacted separately


def test_raw_log_is_not_stored_only_a_digest(tmp_path):
    svc = service(tmp_path)
    record = svc.investigate(req())
    assert len(record.log_stats.sha256) == 64 and record.log_stats.lines == 5
    assert "req-4" not in record.model_dump_json().replace("E4", "")  # req-4 is a non-cited line


def test_record_is_persisted_and_retrievable(tmp_path):
    svc = service(tmp_path)
    record = svc.investigate(req(metadata={"service": "checkout"}))
    saved = svc.store.get(record.id)
    assert saved.status == "completed" and saved.metadata == {"service": "checkout"}
    assert saved.duration_ms >= 0 and saved.tool_history[0].tool == "summarize_errors"


def test_internal_crash_is_contained_and_saved_without_stack_trace(tmp_path):
    import sentinel_ai.service as mod

    original = mod.run_investigation
    mod.run_investigation = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("secret detail"))
    try:
        record = service(tmp_path).investigate(req())
    finally:
        mod.run_investigation = original
    assert record.status == "failed" and record.error == "internal error (RuntimeError)"
    assert "secret detail" not in record.model_dump_json()
