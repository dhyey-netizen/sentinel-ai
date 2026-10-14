"""LLM access. The agent depends only on the ``LLMClient`` protocol, so tests inject fakes."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Protocol

from sentinel_ai.config import Settings


class LLMError(RuntimeError):
    """The model was unreachable, timed out, or returned something that is not a JSON object."""


class LLMClient(Protocol):
    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """Return one parsed JSON object. Raise ``LLMError`` on any failure."""
        ...


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def parse_json_object(raw: str) -> dict[str, Any]:
    """Extract a JSON object from model text (strips qwen3 <think> blocks and code fences)."""
    text = _THINK_RE.sub("", raw).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LLMError("model output contained no JSON object") from None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"model output was not valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise LLMError("model output JSON was not an object")
    return value


class OllamaLLM:
    """qwen3:4b (or any Ollama model) via ``langchain-ollama`` with JSON-constrained output."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._chat: Any = None

    def _client(self) -> Any:
        if self._chat is None:
            from langchain_ollama import ChatOllama  # imported lazily: heavy, optional in tests

            s = self._settings
            self._chat = ChatOllama(
                model=s.ollama_model,
                base_url=s.ollama_base_url,
                temperature=s.ollama_temperature,
                num_ctx=s.ollama_num_ctx,
                format="json",
                client_kwargs={"timeout": s.ollama_timeout_s},
            )
        return self._chat

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        try:
            # "/no_think" disables qwen3's reasoning trace: faster, and keeps JSON clean.
            response = self._client().invoke(
                [("system", system), ("human", f"{user}\n\n/no_think")]
            )
        except ImportError as exc:
            raise LLMError("langchain-ollama is not installed") from exc
        except Exception as exc:  # noqa: BLE001 - system boundary: httpx/ollama raise many types
            raise LLMError(f"Ollama call failed: {type(exc).__name__}") from exc
        content = response.content if isinstance(response.content, str) else str(response.content)
        return parse_json_object(content)


def check_ollama(settings: Settings, timeout_s: float = 3.0) -> tuple[bool, str]:
    """Report whether the Ollama server is up and the configured model is pulled."""
    url = settings.ollama_base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:  # noqa: S310 - configured URL
            payload = json.load(resp)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return False, f"Ollama not reachable at {settings.ollama_base_url} ({type(exc).__name__})"
    names = {m.get("name", "") for m in payload.get("models", [])}
    wanted = settings.ollama_model
    if wanted in names or f"{wanted}:latest" in names:
        return True, f"Ollama is up and model {wanted!r} is available"
    return False, f"Ollama is up but model {wanted!r} is missing; run: ollama pull {wanted}"
