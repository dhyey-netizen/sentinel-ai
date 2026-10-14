"""FastAPI service. Run with ``sentinel serve`` or ``uvicorn --factory sentinel_ai.api:get_app``."""

from __future__ import annotations

import logging
from collections.abc import Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from sentinel_ai import __version__
from sentinel_ai.config import Settings
from sentinel_ai.llm import LLMClient
from sentinel_ai.schemas import InvestigationRecord, InvestigationRequest, InvestigationSummary
from sentinel_ai.service import InputError, InvestigationService
from sentinel_ai.storage import InvestigationStore

log = logging.getLogger("sentinel_ai.api")


def create_app(
    settings: Settings | None = None,
    llm_factory: Callable[[Settings], LLMClient] | None = None,
) -> FastAPI:
    """App factory. ``llm_factory`` lets tests inject a fake model."""
    settings = settings or Settings.from_env()
    store = InvestigationStore(settings.db_path)
    service = InvestigationService(settings, store, llm_factory(settings) if llm_factory else None)

    app = FastAPI(
        title="Sentinel AI",
        version=__version__,
        description="Read-only autonomous incident investigation agent. "
        "Log content is untrusted; no remediation is ever executed.",
    )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error: %s", type(exc).__name__)
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/investigations", response_model=InvestigationRecord, status_code=201)
    def create_investigation(request: InvestigationRequest) -> InvestigationRecord:
        try:
            return service.investigate(request)
        except InputError as exc:
            code = 413 if exc.too_large else 422
            raise HTTPException(status_code=code, detail=str(exc)) from exc

    @app.get("/investigations", response_model=list[InvestigationSummary])
    def list_investigations(limit: int = 20) -> list[InvestigationSummary]:
        return store.list(limit)

    @app.get("/investigations/{investigation_id}", response_model=InvestigationRecord)
    def get_investigation(investigation_id: str) -> InvestigationRecord:
        record = store.get(investigation_id)
        if record is None:
            raise HTTPException(status_code=404, detail="investigation not found")
        return record

    return app


def get_app() -> FastAPI:
    """Factory entry point: ``uvicorn --factory sentinel_ai.api:get_app``."""
    return create_app()
