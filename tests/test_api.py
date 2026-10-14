"""API tests. Require fastapi + httpx (dev extras) and langgraph; skipped when absent."""

import pytest

from helpers import POOL_LOG, ScriptedLLM, good_analysis, settings

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("langgraph")

from fastapi.testclient import TestClient  # noqa: E402

from sentinel_ai.api import create_app  # noqa: E402


def client(tmp_path, **cfg) -> TestClient:
    s = settings(db_path=tmp_path / "api.db", **cfg)
    app = create_app(s, lambda _: ScriptedLLM(analysis=good_analysis(["E2", "E3"])))
    return TestClient(app, raise_server_exceptions=False)


def body(**over):
    return {"question": "Why are checkouts failing?", "log_text": POOL_LOG, **over}


def test_health(tmp_path):
    r = client(tmp_path).get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_create_then_fetch(tmp_path):
    c = client(tmp_path)
    r = c.post("/investigations", json=body(metadata={"service": "checkout"}))
    assert r.status_code == 201
    data = r.json()
    assert data["status"] == "completed" and data["report"]["root_cause"]["title"]
    assert data["report"]["remediation_requires_human_approval"] is True
    got = c.get(f"/investigations/{data['id']}")
    assert got.status_code == 200 and got.json()["id"] == data["id"]
    assert c.get("/investigations").json()[0]["id"] == data["id"]


def test_unknown_id_is_404(tmp_path):
    r = client(tmp_path).get("/investigations/nope")
    assert r.status_code == 404 and r.json() == {"detail": "investigation not found"}


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "", "log_text": "x"},
        {"question": "why?", "log_text": "   "},
        {"question": "why?"},
        {"question": "why?", "log_text": "x", "extra": 1},
    ],
)
def test_invalid_bodies_are_422(tmp_path, payload):
    assert client(tmp_path).post("/investigations", json=payload).status_code == 422


def test_oversized_log_is_413_with_clear_message(tmp_path):
    r = client(tmp_path, max_log_bytes=20).post("/investigations", json=body())
    assert r.status_code == 413 and "limit" in r.json()["detail"]


def test_unhandled_errors_do_not_leak_internals(tmp_path):
    c = client(tmp_path)
    c.app.state  # noqa: B018 - app exists
    import sentinel_ai.storage as storage

    original = storage.InvestigationStore.get
    storage.InvestigationStore.get = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("db path /secret")
    )
    try:
        r = c.get("/investigations/abc")
    finally:
        storage.InvestigationStore.get = original
    assert r.status_code == 500 and r.json() == {"detail": "internal server error"}
    assert "secret" not in r.text
