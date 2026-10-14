# Architecture

## Module map

| Module | Responsibility |
|---|---|
| `config.py` | `Settings` from `SENTINEL_*` env vars, validated |
| `redaction.py` | Regex secret/PII redaction, returns counts (never values) |
| `safety.py` | Injection-likeness flagging, destructive-action filter |
| `schemas.py` | All Pydantic models: input, tool results, evidence, LLM drafts, report, records |
| `diagnostics.py` | `LogCorpus`: parsing + the five deterministic tools |
| `tools.py` | Allow-list registry: tool name → validated args model → corpus method |
| `llm.py` | `LLMClient` protocol, `OllamaLLM`, JSON extraction, `check_ollama` |
| `agent.py` | `Investigator` (graph nodes as methods), routing, `build_graph` |
| `service.py` | validate → redact → run graph → persist; shared by CLI and API |
| `storage.py` | SQLite repository |
| `api.py` / `cli.py` | Interfaces |
| `evaluation.py` | Deterministic eval harness + no-LLM baseline |

## Data flow

1. Request is validated by Pydantic (`InvestigationRequest`), then size limits (`service.validate_limits`).
   Over-limit input is **rejected** with a message, never truncated.
2. Question and logs are redacted. Only the redacted text is parsed into a `LogCorpus`.
   Only a SHA-256, byte/line counts and redaction counts are kept about the raw input.
3. The graph runs against that corpus. The LLM sees only: the question, the tool catalogue,
   compact evidence rows, and previous tool names/arguments, always inside `<log_data>` markers.
4. The report and a compact tool history are saved to SQLite.

## Graph state

`AgentState` (TypedDict) holds: `question`, `evidence` (id → `Evidence`, id = `E<line_no>`),
`tool_history` (audit records), `seen_calls` (dedupe), `error_groups`, `iteration`, `choice`
(pending `ToolChoice`), `hint` (feedback to the planner), `degraded`, `review_rounds`, `notes`
(disclosures), `analysis`, `report`, `status`, `error`.

Nodes return **partial updates** and are methods on `Investigator`, so each is unit-testable
without LangGraph (`tests/test_agent.py` bottom half).

## Transitions

| From | Condition | To |
|---|---|---|
| intake | no log lines | END (`failed`) |
| intake | ok | initial_evidence |
| plan | `finish`, or planner failed twice | analyze |
| plan | `call_tool` | execute_tool |
| aggregate | `iteration >= max_iterations` | analyze |
| aggregate | otherwise | plan |
| review | top hypothesis has < 2 valid citations, first review, not degraded | plan (once) |
| review | otherwise | report |

**Termination:** `iteration` increments every executed (or rejected) tool step and is checked
against `max_iterations`; the review extension happens at most once (`review_rounds`); the graph is
also invoked with `recursion_limit=60` as a backstop.

## Tool selection and execution

The planner prompt lists tools with their argument names. The model must answer
`{"action","tool","arguments","reason"}`. `ToolRegistry.execute` then:
1. rejects names not in `TOOL_SPECS`;
2. validates arguments with a `extra="forbid"` Pydantic model (types, ranges, `request_id` regex);
3. runs the method on the corpus bound to this investigation;
4. returns a `ToolResult`; its log lines are merged into the evidence map (capped, with disclosure).

Rejected calls are recorded in the audit history and fed back to the planner as a hint, consuming
an iteration. Repeating an identical call is refused as a duplicate.

## Hallucination controls

- Report evidence text comes **from the corpus**, never from model output.
- Cited ids that do not exist are dropped (with a review note).
- Final confidence = `min(model confidence, evidence cap)`: 0 valid citations → insufficient;
  1 → low; correlation-only, contradicted, or < 4 citations → at most medium; degraded run → at most low.
- No valid evidence ⇒ no root cause; `insufficient_evidence` is set for low/insufficient outcomes.
- Prompts require "correlation vs causation" labelling; the field is part of the schema.

## Failure handling

| Failure | Behaviour |
|---|---|
| Ollama unreachable | one attempt (no retry), degraded flag, deterministic report |
| Malformed JSON / schema mismatch | one retry including the error, then degrade |
| Unknown tool / bad args | audited rejection, hint, continue |
| Unexpected exception in the graph | caught at the service boundary; record saved as `failed` with `internal error (<Type>)` only |

## Scaling to many concurrent incidents (design, not implemented)

Investigations are independent and share nothing but the DB. Put `POST /investigations` behind a
queue (e.g. Postgres/Redis-backed workers), return `202` + id, and poll `GET`. The bottleneck is
LLM throughput: batch/queue requests per Ollama instance (or use vLLM), cap concurrency by VRAM,
and move SQLite to Postgres for multi-writer use. Logs would be passed by reference (object store)
rather than in the request body.
