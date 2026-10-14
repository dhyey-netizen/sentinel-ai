"""Allow-listed tool registry. The LLM can only name a tool and supply validated arguments.

There is deliberately no generic "run code / shell / SQL / read file" tool. Each tool is bound
to the LogCorpus of the current investigation, so cross-investigation access is impossible.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from sentinel_ai.diagnostics import LogCorpus, ToolError
from sentinel_ai.schemas import ToolResult


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchLogsArgs(_Args):
    query: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    limit: int = Field(default=20, ge=1, le=50)


class SummarizeErrorsArgs(_Args):
    pass


class TraceRequestArgs(_Args):
    request_id: Annotated[str, StringConstraints(pattern=r"^[\w.-]{3,64}$")]


class FindTimeWindowArgs(_Args):
    start_time: Annotated[str, StringConstraints(min_length=10, max_length=40)]
    end_time: Annotated[str, StringConstraints(min_length=10, max_length=40)]
    limit: int = Field(default=30, ge=1, le=50)


class ExtractErrorContextArgs(_Args):
    error_identifier: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    context_lines: int = Field(default=3, ge=0, le=10)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[_Args]
    run: Callable[[LogCorpus, Any], ToolResult]


TOOL_SPECS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            "search_logs",
            "Find log lines containing ALL whitespace-separated terms (case-insensitive).",
            SearchLogsArgs,
            lambda c, a: c.search_logs(a.query, a.limit),
        ),
        ToolSpec(
            "summarize_errors",
            "Group ERROR/CRITICAL lines into recurring signatures with counts and busiest minutes.",
            SummarizeErrorsArgs,
            lambda c, a: c.summarize_errors(),
        ),
        ToolSpec(
            "trace_request",
            "Return every log line for one request id, in order.",
            TraceRequestArgs,
            lambda c, a: c.trace_request(a.request_id),
        ),
        ToolSpec(
            "find_time_window",
            "Return lines between two ISO-8601 timestamps (most severe first).",
            FindTimeWindowArgs,
            lambda c, a: c.find_time_window(a.start_time, a.end_time, a.limit),
        ),
        ToolSpec(
            "extract_error_context",
            "Show lines surrounding the first occurrences of an error string or exception name.",
            ExtractErrorContextArgs,
            lambda c, a: c.extract_error_context(a.error_identifier, a.context_lines),
        ),
    )
}


class ToolRegistry:
    """Executes allow-listed tools against one corpus."""

    def __init__(self, corpus: LogCorpus) -> None:
        self._corpus = corpus

    @staticmethod
    def describe() -> str:
        """Human/LLM-readable tool catalogue."""
        lines = []
        for spec in TOOL_SPECS.values():
            fields = ", ".join(
                f"{n}: {f.annotation.__name__ if hasattr(f.annotation, '__name__') else 'str'}"
                f"{'' if f.is_required() else ' (optional)'}"
                for n, f in spec.args_model.model_fields.items()
            )
            lines.append(f"- {spec.name}({fields}): {spec.description}")
        return "\n".join(lines)

    def execute(self, name: str, arguments: dict[str, Any]) -> tuple[ToolResult, dict[str, Any]]:
        """Validate and run a tool. Returns ``(result, normalised_arguments)``.

        Raises ``ToolError`` for unknown tools, invalid arguments, or unanswerable queries.
        """
        spec = TOOL_SPECS.get(name)
        if spec is None:
            raise ToolError(f"unknown tool {name!r}; allowed: {', '.join(TOOL_SPECS)}")
        try:
            args = spec.args_model.model_validate(arguments)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
            )
            raise ToolError(f"invalid arguments for {name}: {problems}") from exc
        return spec.run(self._corpus, args), args.model_dump()
