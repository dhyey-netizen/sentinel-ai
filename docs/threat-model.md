# Threat model

**Assets:** the host machine, the SQLite history, secrets that appear in logs, the integrity of the
investigation conclusions.
**Trust boundaries:** log text and the question are *untrusted*; the LLM output is *untrusted*;
local code and configuration are trusted.

## Primary control: nothing dangerous exists to be invoked

| Capability | Status |
|---|---|
| Shell / Python execution | none exposed |
| SQL / DB access by the model | none; storage is written only by the service |
| Filesystem access by the model | none; tools see only the in-memory corpus |
| Network access by the model | none (Ollama call is made by the app, not by a tool) |
| Remediation actions | none in v1; report is advice only |

The model can only name one of five read-only tools and supply arguments validated by strict
Pydantic models (`extra="forbid"`, type/range/regex constraints). Tools are bound to one
investigation's corpus, so cross-investigation reads are impossible.

## Threats and mitigations

| Threat | Mitigation | Residual risk |
|---|---|---|
| **Prompt injection in logs** ("ignore previous instructions…") | Delimited `<log_data>` blocks + system rule; injection-like lines counted and disclosed in the report; model cannot act beyond read-only tools; destructive suggestions filtered from output | A crafted line can still bias a small model's *analysis text* or hypothesis ranking. Treat reports as advice. |
| **Injection via the question field** | Redacted, length-limited, same delimiting rules apply to evidence (question is user-supplied by design) | Question is trusted as the operator's intent |
| **Secrets in logs leaking to model/DB/API** | Redaction before parsing; raw logs never stored; audit logging records ids/counts only | Regex redaction is best-effort; unusual formats can slip through |
| **Destructive advice followed blindly** | Pattern filter removes e.g. `rm -rf`, `DROP TABLE`, `curl … \| sh`; remediation labelled UNVERIFIED; `remediation_requires_human_approval` fixed to `true` in schema | Filter is a deny-list; harmful advice phrased differently passes. Human review is the real control. |
| **Resource exhaustion** | Byte, question, line, evidence and iteration limits; 413 on oversized input; LLM timeout | No rate limiting or auth on the API |
| **Information disclosure via errors** | API returns generic 500; internal exceptions stored as `internal error (<Type>)` | — |
| **SQL injection** | Parameterised queries only; test covers hostile ids | — |
| **Tool argument abuse** (path traversal, huge limits) | `request_id` regex, numeric bounds, extra-field rejection | — |
| **Fabricated evidence** | Report excerpts come from the corpus by id; unknown ids dropped | Cited-but-irrelevant lines are not semantically checked |

## Not in scope / known gaps

- API has no authentication; run on localhost or behind a reverse proxy with auth.
- The SQLite file stores redacted excerpts and questions; protect it like the logs themselves.
- No defence against a malicious *Ollama server* or model weights.
- Future remediation must be a separate, human-approved tool tier with its own audit trail; do not
  add write tools to the current registry.
