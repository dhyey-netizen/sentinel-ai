"""Command line interface: investigate / history / show / serve / doctor / eval."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# pyrefly: ignore [missing-import]
from pydantic import ValidationError

from sentinel_ai.config import ConfigError, Settings
from sentinel_ai.schemas import InvestigationRecord, InvestigationRequest, Report
from sentinel_ai.service import InputError, InvestigationService
from sentinel_ai.storage import InvestigationStore


def render_report(report: Report) -> str:
    out: list[str] = []
    add = out.append

    def section(title: str, items: list[str]) -> None:
        add(f"\n{title}")
        if items:
            out.extend(f"  - {i}" for i in items)
        else:
            add("  (none)")

    add(f"INCIDENT SUMMARY\n  {report.incident_summary}")
    section("OBSERVED SYMPTOMS", report.observed_symptoms)
    add("\nEVIDENCE (excerpts from the supplied logs, redacted)")
    for ev in report.evidence:
        add(f"  {ev.id:<7} line {ev.line_no:<5} {ev.text}")
    add("\nMOST LIKELY ROOT CAUSE")
    if report.root_cause:
        rc = report.root_cause
        add(f"  {rc.title}\n  {rc.explanation}")
        add(
            f"  supporting: {', '.join(rc.evidence_ids) or 'none'}  "
            f"causation: {rc.causation.replace('_', ' ')}"
        )
        for n in rc.review_notes:
            add(f"  review: {n}")
    else:
        add("  Not established - insufficient evidence.")
    add("\nALTERNATIVE HYPOTHESES")
    for h in report.alternative_hypotheses or []:
        add(f"  - {h.title} [{h.confidence}] evidence: {', '.join(h.evidence_ids) or 'none'}")
    if not report.alternative_hypotheses:
        add("  (none)")
    add(f"\nCONFIDENCE: {report.confidence.upper()}\n  {report.confidence_explanation}")
    section("RECOMMENDED NEXT DIAGNOSTIC STEPS (read-only)", report.recommended_next_steps)
    section(
        "SUGGESTED REMEDIATION (UNVERIFIED - requires human approval, never auto-run)",
        report.suggested_remediation,
    )
    section("UNKNOWNS AND LIMITATIONS", report.unknowns_and_limitations)
    return "\n".join(out)


def render_record(record: InvestigationRecord) -> str:
    head = (
        f"Investigation {record.id}  [{record.status}]  {record.created_at:%Y-%m-%d %H:%M:%S}Z  "
        f"{record.duration_ms} ms\nQuestion: {record.question}"
    )
    parts = [head]
    if record.log_stats:
        red = ", ".join(f"{k}={v}" for k, v in record.log_stats.redactions.items()) or "none"
        parts.append(
            f"Logs: {record.log_stats.lines} lines, {record.log_stats.bytes} bytes, "
            f"redactions: {red}"
        )
    if record.error:
        parts.append(f"Error: {record.error}")
    if record.report:
        parts.append(render_report(record.report))
    if record.tool_history:
        parts.append("\nTOOL CALLS")
        for t in record.tool_history:
            state = "ok" if t.ok else f"REJECTED ({t.error})"
            parts.append(f"  {t.step}. {t.tool}({t.arguments}) {state} [{t.duration_ms} ms]")
    return "\n".join(parts)


def _parse_metadata(pairs: list[str]) -> dict[str, str]:
    meta: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise InputError(f"metadata must look like key=value, got {pair!r}")
        meta[key] = value
    return meta


def _read_log_file(path: Path, settings: Settings) -> str:
    if not path.is_file():
        raise InputError(f"log file not found: {path}")
    size = path.stat().st_size
    if size > settings.max_log_bytes:
        raise InputError(
            f"{path} is {size} bytes; the limit is {settings.max_log_bytes} "
            "(set SENTINEL_MAX_LOG_BYTES to raise it)",
            too_large=True,
        )
    return path.read_text(encoding="utf-8", errors="replace")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sentinel", description="Sentinel AI incident investigator")
    sub = p.add_subparsers(dest="command", required=True)

    inv = sub.add_parser("investigate", help="investigate a log file")
    inv.add_argument("--file", required=True, type=Path)
    inv.add_argument("--question", required=True)
    inv.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE")
    inv.add_argument("--json", action="store_true", help="print the full record as JSON")

    hist = sub.add_parser("history", help="list saved investigations")
    hist.add_argument("--limit", type=int, default=20)

    show = sub.add_parser("show", help="show a saved investigation")
    show.add_argument("investigation_id")
    show.add_argument("--json", action="store_true")

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    sub.add_parser("doctor", help="check that Ollama and the model are available")

    ev = sub.add_parser("eval", help="run the deterministic evaluation suite")
    ev.add_argument("--cases", type=Path, default=Path("data/evaluation/cases.json"))
    ev.add_argument("--live", action="store_true", help="use Ollama instead of the no-LLM baseline")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        settings = Settings.from_env()
        return _dispatch(args, settings)
    except (ConfigError, InputError, ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace, settings: Settings) -> int:
    if args.command == "investigate":
        request = InvestigationRequest(
            question=args.question,
            log_text=_read_log_file(args.file, settings),
            metadata=_parse_metadata(args.meta),
        )
        service = InvestigationService(settings, InvestigationStore(settings.db_path))
        record = service.investigate(request)
        print(record.model_dump_json(indent=2) if args.json else render_record(record))
        return 0 if record.status != "failed" else 1

    store = InvestigationStore(settings.db_path)
    if args.command == "history":
        rows = store.list(args.limit)
        if not rows:
            print("no saved investigations")
        for r in rows:
            print(
                f"{r.id}  {r.created_at:%Y-%m-%d %H:%M}  {r.status:<19} {r.duration_ms:>6} ms  "
                f"{r.question[:60]}"
            )
        return 0

    if args.command == "show":
        record = store.get(args.investigation_id)
        if record is None:
            print(f"error: no investigation with id {args.investigation_id!r}", file=sys.stderr)
            return 1
        print(record.model_dump_json(indent=2) if args.json else render_record(record))
        return 0

    if args.command == "serve":
        # pyrefly: ignore [missing-import]
        import uvicorn

        uvicorn.run("sentinel_ai.api:get_app", factory=True, host=args.host, port=args.port)
        return 0

    if args.command == "doctor":
        from sentinel_ai.llm import check_ollama

        ok, message = check_ollama(settings)
        print(message)
        return 0 if ok else 1

    if args.command == "eval":
        from sentinel_ai.evaluation import UnavailableLLM, format_results, run_evaluation
        from sentinel_ai.llm import OllamaLLM

        llm = OllamaLLM(settings) if args.live else UnavailableLLM()
        label = f"live: {settings.ollama_model}" if args.live else "baseline: no LLM"
        results = run_evaluation(args.cases, settings, llm)
        print(format_results(results, label))
        return 0 if all(r.passed for r in results) else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
