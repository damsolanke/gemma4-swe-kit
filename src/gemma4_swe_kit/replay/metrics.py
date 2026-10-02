"""Metrics for replayed samples. A metric takes a Sample and returns a bool or a number; booleans are reported
as rates. Custom metrics: ``--metric package.module:function``.
"""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from ..toolcalls import (DAMAGED_ARGS_RE, TOOL_CALL_START, call_name_candidates, canonical_call, classify_calls,
                         declared_tool_names)


@dataclass
class Sample:
    body: dict                       # the request that was sent (after the condition's changes)
    message: dict                    # the assistant message returned
    finish: str
    raw: str | None = None           # raw generation, when the server returns it (g4kit-proxy does)
    usage: dict = field(default_factory=dict)
    prev_call: tuple[str, str] | None = None

    @property
    def declared(self) -> set[str]:
        return declared_tool_names(self.body.get("tools"))

    @property
    def calls(self) -> list[tuple[str, Any]]:
        out = []
        for tc in self.message.get("tool_calls") or []:
            fn = tc.get("function", {})
            out.append((fn.get("name", ""), fn.get("arguments")))
        return out

    @property
    def content(self) -> str:
        return self.message.get("content") or ""


def malformed_name(s: Sample) -> bool:
    """Something other than a declared tool name sits in a call's name slot (e.g. a shell command)."""
    if s.raw is not None:
        return bool(classify_calls(s.raw, s.declared)["malformed_names"])
    names = call_name_candidates(s.content) + [name for name, _ in s.calls]
    return any(n not in s.declared for n in names)


def unparsed_call(s: Sample) -> bool:
    """A call vLLM's pattern does not extract: the harness answers with its 'token limit' nudge."""
    return TOOL_CALL_START in s.content and not s.calls


def unknown_tool(s: Sample) -> bool:
    """A well-formed call to an undeclared tool: ADK raises and the task loses its patch."""
    return any(name not in s.declared for name, _ in s.calls)


def no_tool_call(s: Sample) -> bool:
    return not s.calls


def text_only(s: Sample) -> bool:
    return not s.calls and bool(s.content.strip()) and TOOL_CALL_START not in s.content


def exact_repeat(s: Sample) -> bool:
    """The first call repeats the context's previous call exactly (name and arguments)."""
    if not s.calls or s.prev_call is None:
        return False
    return canonical_call(*s.calls[0]) == s.prev_call


def damaged_args(s: Sample) -> bool:
    """Backtick-closed strings or kwarg syntax inside a call."""
    text = s.raw if s.raw is not None else s.content + " ".join(json.dumps(a) if not isinstance(a, str) else a
                                                               for _, a in s.calls)
    return bool(DAMAGED_ARGS_RE.search(text or ""))


def completion_tokens(s: Sample) -> float:
    return float(s.usage.get("completion_tokens") or 0)


def finish_length(s: Sample) -> bool:
    return s.finish == "length"


BUILTIN: dict[str, Callable[[Sample], Any]] = {
    "malformed_name": malformed_name, "unparsed_call": unparsed_call, "unknown_tool": unknown_tool,
    "no_tool_call": no_tool_call, "text_only": text_only, "exact_repeat": exact_repeat,
    "damaged_args": damaged_args, "completion_tokens": completion_tokens, "finish_length": finish_length,
}
DEFAULT = ["malformed_name", "unparsed_call", "unknown_tool", "text_only", "exact_repeat", "completion_tokens"]


def resolve(names: list[str] | None) -> dict[str, Callable[[Sample], Any]]:
    out = {}
    for name in names or DEFAULT:
        if name in BUILTIN:
            out[name] = BUILTIN[name]
        elif ":" in name:
            mod, _, attr = name.partition(":")
            out[attr] = getattr(importlib.import_module(mod), attr)
        else:
            raise ValueError(f"unknown metric {name!r}; built-in: {', '.join(BUILTIN)}; or module:function")
    return out
