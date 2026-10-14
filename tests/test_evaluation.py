from pathlib import Path

import pytest

from helpers import settings
from sentinel_ai.evaluation import (
    UnavailableLLM,
    check_case,
    format_results,
    load_cases,
    run_evaluation,
)

CASES = Path(__file__).resolve().parent.parent / "data" / "evaluation" / "cases.json"


def test_dataset_is_well_formed():
    cases = load_cases(CASES)
    assert len(cases) >= 7 and len({c["id"] for c in cases}) == len(cases)
    for c in cases:
        assert (CASES.parent / c["log_file"]).is_file()


def test_baseline_runs_every_case_without_failing_status():
    pytest.importorskip("langgraph")
    results = run_evaluation(CASES, settings(), UnavailableLLM())
    assert len(results) == len(load_cases(CASES))
    assert all(r.status == "completed_degraded" for r in results)
    assert all(r.passed for r in results), format_results(results, "baseline")


def test_checks_can_actually_fail():
    """Guards against a vacuous harness: tamper with expectations and the checks must fail."""
    pytest.importorskip("langgraph")
    from sentinel_ai.schemas import InvestigationRequest
    from sentinel_ai.service import InvestigationService
    from sentinel_ai.storage import InvestigationStore

    case = next(c for c in load_cases(CASES) if c["id"] == "pool_exhaustion")
    svc = InvestigationService(settings(), InvestigationStore(":memory:"), UnavailableLLM())
    record = svc.investigate(
        InvestigationRequest(
            question=case["question"], log_text=(CASES.parent / case["log_file"]).read_text()
        )
    )
    tampered = {
        **case,
        "expected_tools": ["find_time_window"],
        "expected_evidence_substrings": ["never appears"],
        "root_cause_keywords_any": ["kubernetes"],
        "expect_insufficient": True,
        "max_confidence": "insufficient",
        "forbidden_in_actions": ["review"],
        "expect_injection_flagged": True,
    }
    result = check_case(tampered, record)
    assert result.checks["tools"] is False
    assert result.checks["evidence"] is False
    assert result.checks["rootcause"] is False
    assert result.checks["uncertainty"] is False
    assert result.checks["safety"] is False
