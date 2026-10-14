# Interview guide

Answers refer to the actual code. Be ready to say what was and wasn't verified (see README).

**Why LangGraph rather than a simple chain?**
The control flow has a loop (plan ⇄ execute), a conditional exit (finish / iteration limit /
model failure) and a review step that can send work back once. A chain can't express that cleanly.
LangGraph gives explicit state and named transitions (`agent.build_graph`). Honest caveat: the
nodes are plain methods, so the logic is testable without it; LangGraph is the wiring.

**What makes this agentic?**
The model decides *which* diagnostic to run next based on evidence so far, observes the result,
and decides again, until it finishes or hits the bound. A deterministic first pass
(`initial_evidence`) gives it a grounded starting point; the review node can make it gather more.

**How is state managed?**
One `AgentState` TypedDict; nodes return partial updates. Evidence is a dict keyed by `E<line>`,
so merging tool results is idempotent and citations are stable. Tool history doubles as the audit log.

**How are tools selected and executed?**
LLM emits JSON `{action, tool, arguments}`. `ToolRegistry.execute` allow-lists the name, validates
arguments with strict Pydantic models, runs a method on the investigation's own `LogCorpus`.
Failures are audited and fed back as a hint. Duplicates are refused.

**How do you control hallucinations?**
(1) Evidence text in the report is copied from the corpus, never from model output. (2) Cited ids
that don't exist are dropped. (3) Confidence is `min(model, evidence cap)`; the model can't
self-certify "high". (4) No valid citation ⇒ no root cause. (5) Correlation vs causation is a
schema field. Limitation: we check that cited lines *exist*, not that they *support* the claim.

**How do you evaluate evidence quality?**
Deterministically in `evaluation.py`: expected tools used, expected log text retrieved, citations
resolve, root-cause keywords, uncertainty flags/caps, safety strings. Seven synthetic cases
including contradictory, undeterminable and injected logs. No LLM judge. Limitation: small,
synthetic, lenient keyword checks; the no-LLM baseline measures the pipeline floor, not model skill.

**What happens when the model fails?**
Transport failure: no retry, `degraded=True`. Bad JSON/schema: one retry with the error message,
then degrade. Degraded runs still return a labelled deterministic report (most frequent error,
low confidence cap, explicit disclosure). Tested in `test_agent.py`.

**How do you prevent prompt injection from logs?**
Mostly *structurally*: there are no dangerous tools, so an obeyed instruction can do little.
Additionally: delimited untrusted-data blocks, system rule, flagging of instruction-like lines,
destructive-output filter, human-approval marker. Be explicit that this reduces but does not
eliminate the risk of biased analysis text (threat-model.md).

**Why is the first version read-only?**
Wrong remediation is high-impact and hard to reverse; the model is a 4B local model and
prompt-injectable. Read-only lets us ship something safe, and a future write tier would need
approval gates and separate audit — not new tools bolted into this registry.

**How would you scale to many concurrent incidents?**
Queue + workers, `202` + polling, LLM concurrency bounded by VRAM (or vLLM batching), Postgres
instead of SQLite, logs by reference. Not implemented; see architecture.md.

## Questions to expect about the code

- *Why cap confidence deterministically?* Small models are overconfident; calibration by evidence
  count is crude but auditable.
- *Why `E<line_no>` ids?* Stable across tools, lets the model cite without copying text.
- *Why JSON mode instead of native tool calls?* Portability across Ollama models and one validation path.
- *What would you change first?* Measure `--live` on real incidents, add a semantic support check.

## Resume bullets — use only what you have verified

Safe once you have run `pytest` locally and seen it pass:
- Built a read-only incident-investigation agent in Python (LangGraph, Ollama/qwen3:4b, FastAPI,
  SQLite) with a bounded tool loop, deterministic log-diagnostic tools and Pydantic-validated reports.
- Designed evidence gating (citation checking, deterministic confidence caps, degraded-mode
  fallback) and a deterministic evaluation suite of 7 synthetic incidents incl. prompt-injection cases.

Add **only after you run it**: live latency/VRAM numbers, `sentinel eval --live` pass counts.
Do not claim production use, accuracy, or benchmark results you have not measured.
