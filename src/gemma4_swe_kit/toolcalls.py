"""Gemma 4 output parsing that matches the scorer's vLLM 0.19.1 server (``--reasoning-parser gemma4
--tool-call-parser gemma4 --enable-auto-tool-choice``), plus malformed-call detection.

Gemma 4 writes calls as ``<|tool_call>call:NAME{key:<|"|>text<|"|>,n:3}<tool_call|>`` and thoughts as
``<|channel>thought\\n...<channel|>``. vLLM extracts a call only when the whole call matches
``<\\|tool_call>call:([\\w\\-\\.]+)\\{(.*?)\\}<tool_call\\|>``. Anything else stays in the message content, which is
what the competition harness inspects: content containing ``<|tool_call>`` triggers the "token limit" nudge,
and three nudges in a row end the session, so the task keeps only edits already on disk. A well-formed call to an undeclared name is
extracted, and ADK then raises on the unknown tool, which the harness turns into an empty patch.
"""
from __future__ import annotations

import json
import re
import uuid
import warnings
from pathlib import Path
from typing import Any, Callable

TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"
STRING_DELIM = '<|"|>'
CHANNEL_START = "<|channel>"
CHANNEL_END = "<channel|>"
THOUGHT_PREFIX = "thought\n"
EMPTY_THOUGHT = CHANNEL_START + THOUGHT_PREFIX + CHANNEL_END
# generation_config.json eos_token_id = [1, 106, 50]: <eos>, <turn|>, <|tool_response>
STOP_STRINGS = ("<eos>", "<turn|>", "<|tool_response>")

TOOL_CALL_RE = re.compile(r"<\|tool_call>call:([\w\-\.]+)\{(.*?)\}<tool_call\|>", re.DOTALL)
# lenient scan for anything written in the tool-name slot, including shell commands
CALL_NAME_RE = re.compile(r"<\|tool_call>call:(.*?)(?=\{|<\|\"\|>|<tool_call\|>)", re.DOTALL)
# argument damage seen in practice: backtick-closed strings, kwarg syntax
DAMAGED_ARGS_RE = re.compile(r'`,\w+:|\{\w+="|\w+=<\|"\|>')

ArgsParser = Callable[[str], dict]


class ParserHang(RuntimeError):
    """vLLM 0.19.1's gemma4 argument parser never returns on this input: a bare ']' inside an array whose
    brackets were matched across a string loops forever in ``_parse_gemma4_array``. The built-in parser detects
    the condition and raises instead of looping."""


# ---------------------------------------------------------------------------------------------------------
# built-in argument parser: an independent implementation of the format parsed by vLLM's gemma4 parser
# ---------------------------------------------------------------------------------------------------------

def _scalar(text: str) -> Any:
    text = text.strip()
    if not text:
        return text
    if text == "true":
        return True
    if text == "false":
        return False
    if text.lower() in ("null", "none", "nil"):
        return None
    try:
        return float(text) if "." in text else int(text)
    except ValueError:
        return text


def _skip_string(s: str, i: int) -> int:
    """i points at an opening delimiter; return the index just past the closing one (or len(s))."""
    i += len(STRING_DELIM)
    close = s.find(STRING_DELIM, i)
    return len(s) if close == -1 else close + len(STRING_DELIM)


def _match_braces(s: str, i: int, open_ch: str, close_ch: str, skip_strings: bool) -> tuple[int, int]:
    """i points just past an opening bracket. Return (end index, remaining depth)."""
    depth = 1
    n = len(s)
    while i < n and depth > 0:
        if skip_strings and s.startswith(STRING_DELIM, i):
            i = _skip_string(s, i)
            continue
        if s[i] == open_ch:
            depth += 1
        elif s[i] == close_ch:
            depth -= 1
        i += 1
    return i, depth


def builtin_parse_args(s: str, partial: bool = False) -> dict:
    """Parse ``key:value,...`` into a dict. Keys run up to the first ':'; text without one is dropped."""
    if not s or not s.strip():
        return {}
    out: dict = {}
    i, n = 0, len(s)
    while i < n:
        while i < n and s[i] in " ,\n\t":
            i += 1
        if i >= n:
            break
        colon = s.find(":", i)
        if colon == -1:
            break
        key = s[i:colon].strip()
        i = colon + 1
        while i < n and s[i] in " \n\t":
            i += 1
        if i >= n:
            if not partial:
                out[key] = ""
            break
        if s.startswith(STRING_DELIM, i):
            start = i + len(STRING_DELIM)
            close = s.find(STRING_DELIM, start)
            if close == -1:
                out[key] = s[start:]
                break
            out[key] = s[start:close]
            i = close + len(STRING_DELIM)
        elif s[i] == "{":
            end, depth = _match_braces(s, i + 1, "{", "}", skip_strings=True)
            out[key] = (builtin_parse_args(s[i + 1:end], partial=True) if depth > 0
                        else builtin_parse_args(s[i + 1:end - 1]))
            i = end
        elif s[i] == "[":
            end, depth = _match_braces(s, i + 1, "[", "]", skip_strings=True)
            out[key] = (_parse_array(s[i + 1:end], partial=True) if depth > 0
                        else _parse_array(s[i + 1:end - 1]))
            i = end
        else:
            start = i
            while i < n and s[i] not in ",}]":
                i += 1
            if partial and i >= n:
                break
            out[key] = _scalar(s[start:i])
    return out


def _parse_array(s: str, partial: bool = False) -> list:
    items: list = []
    i, n = 0, len(s)
    while i < n:
        while i < n and s[i] in " ,\n\t":
            i += 1
        if i >= n:
            break
        if s.startswith(STRING_DELIM, i):
            start = i + len(STRING_DELIM)
            close = s.find(STRING_DELIM, start)
            if close == -1:
                items.append(s[start:])
                break
            items.append(s[start:close])
            i = close + len(STRING_DELIM)
        elif s[i] == "{":
            end, depth = _match_braces(s, i + 1, "{", "}", skip_strings=True)
            items.append(builtin_parse_args(s[i + 1:end], partial=True) if depth > 0
                         else builtin_parse_args(s[i + 1:end - 1]))
            i = end
        elif s[i] == "[":
            # vLLM counts brackets of nested arrays without skipping string contents; kept for parity
            end, depth = _match_braces(s, i + 1, "[", "]", skip_strings=False)
            items.append(_parse_array(s[i + 1:end], partial=True) if depth > 0 else _parse_array(s[i + 1:end - 1]))
            i = end
        else:
            start = i
            while i < n and s[i] not in ",]":
                i += 1
            if partial and i >= n:
                break
            if i == start and i < n and s[i] == "]":
                raise ParserHang(s)
            items.append(_scalar(s[start:i]))
    return items


def load_vllm_args_parser(path: str | Path) -> ArgsParser:
    """Load ``_parse_gemma4_args`` from vLLM's gemma4_tool_parser.py without importing vLLM.

    Only the module's pure-Python helper functions (between ``def _parse_gemma4_value`` and
    ``class Gemma4ToolParser``) are executed.
    """
    src = Path(path).read_text(encoding="utf-8")
    start, end = src.index("def _parse_gemma4_value"), src.index("class Gemma4ToolParser")
    ns: dict[str, Any] = {"json": json, "STRING_DELIM": STRING_DELIM}
    exec(compile(src[start:end], str(path), "exec"), ns)
    fn = ns["_parse_gemma4_args"]

    def parse(s: str) -> dict:
        builtin_parse_args(s)   # raises ParserHang where vLLM's own function would never return
        return fn(s)

    return parse


def load_args_parser(path: str | Path | None = None, *, quiet: bool = False) -> tuple[ArgsParser, str]:
    """vLLM's own parser when its file can be found (see assets.find_parser), else the built-in one."""
    from .assets import find_parser

    found = find_parser(path)
    if found is not None:
        try:
            return load_vllm_args_parser(found), f"vllm:{found}"
        except Exception as exc:   # unexpected file layout
            if not quiet:
                warnings.warn(f"could not load vLLM's gemma4 parser from {found}: {exc}; using the built-in parser",
                              stacklevel=2)
    return builtin_parse_args, "builtin"


# ---------------------------------------------------------------------------------------------------------
# reasoning and tool-call extraction (non-streaming path of vLLM 0.19.1's chat completions)
# ---------------------------------------------------------------------------------------------------------

def strip_stop(raw: str) -> str:
    """Remove one trailing stop token, which vLLM never includes in the output text."""
    for stop in STOP_STRINGS:
        if raw.endswith(stop):
            return raw[: -len(stop)]
    return raw


def extract_reasoning(text: str) -> tuple[str | None, str | None]:
    """Gemma4ReasoningParser.extract_reasoning: (reasoning, content).

    Text before ``<|channel>`` is dropped; an unterminated thought is all reasoning (content None); the
    ``thought\\n`` channel label is removed from the reasoning.
    """
    if CHANNEL_START not in text and CHANNEL_END not in text:
        return None, text
    before, sep, after = text.partition(CHANNEL_START)
    rest = after if sep else before
    if CHANNEL_END not in rest:
        reasoning, content = rest, None
    else:
        reasoning, _, content = rest.partition(CHANNEL_END)
        content = content or None
    if reasoning.startswith(THOUGHT_PREFIX):
        reasoning = reasoning[len(THOUGHT_PREFIX):]
    return reasoning, content


def extract_tool_calls(content: str | None, parse_args: ArgsParser) -> tuple[list[tuple[str, dict]], str | None]:
    """Gemma4ToolParser.extract_tool_calls: ([(name, args)], content left for the message)."""
    text = content if content is not None else ""
    if TOOL_CALL_START not in text:
        return [], content
    matches = TOOL_CALL_RE.findall(text)
    if not matches:
        return [], content
    try:
        calls = [(name, parse_args(args)) for name, args in matches]
    except ParserHang:
        raise
    except Exception:   # vLLM logs the exception and returns the output as content
        return [], content
    idx = text.find(TOOL_CALL_START)
    before = text[:idx].strip() if idx > 0 else None
    return calls, (before or None)


def new_call_id() -> str:
    return "chatcmpl-tool-" + uuid.uuid4().hex


def parse_completion(raw: str, *, parse_args: ArgsParser, finish_reason: str = "stop", tools_present: bool = True,
                     tool_choice: Any = None, include_reasoning: bool = True) -> tuple[dict, str]:
    """Build the assistant message and finish_reason vLLM returns for a raw generation."""
    reasoning, content = extract_reasoning(strip_stop(raw))
    if not include_reasoning:
        reasoning = None
    calls: list[tuple[str, dict]] = []
    if tool_choice != "none":   # "auto", unset, or anything else (treated as auto)
        calls, content = extract_tool_calls(content, parse_args)
    message: dict[str, Any] = {"role": "assistant", "content": content, "reasoning": reasoning, "tool_calls": []}
    if calls and tools_present:
        message["tool_calls"] = [{"id": new_call_id(), "type": "function",
                                  "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}
                                 for name, args in calls]
        finish_reason = "tool_calls"
    return message, finish_reason


# ---------------------------------------------------------------------------------------------------------
# malformed-call detection
# ---------------------------------------------------------------------------------------------------------

def call_name_candidates(text: str) -> list[str]:
    """Whatever the model wrote in the tool-name slot of each call, e.g. 'run_command' or 'grep -rn foo .'."""
    return [m.group(1).strip() for m in CALL_NAME_RE.finditer(text or "")]


def declared_tool_names(tools: list[dict] | None) -> set[str]:
    return {t.get("function", {}).get("name") for t in tools or [] if t.get("function", {}).get("name")}


def classify_calls(text: str, declared: set[str] | None) -> dict[str, Any]:
    """Classify the calls in a raw generation (or in returned content, which keeps unparsed calls).

    malformed_names: names from the lenient scan that are not declared tools (shell commands, kwarg-style
                     names); this is the replay metric used for the prose-instruction experiment.
    unparsed:        the text holds ``<|tool_call>`` but vLLM's pattern matches nothing (harness nudge).
    unknown_tools:   well-formed calls whose name is not declared (ADK raises; the task loses its patch).
    """
    text = text or ""
    declared = declared or set()
    names = call_name_candidates(text)
    parsed = [name for name, _ in TOOL_CALL_RE.findall(text)]
    return {
        "names": names,
        "malformed_names": [n for n in names if n not in declared] if declared else [],
        "unparsed": TOOL_CALL_START in text and not parsed,
        "unknown_tools": [n for n in parsed if declared and n not in declared],
        "damaged_args": bool(DAMAGED_ARGS_RE.search(text)),
    }


def canonical_call(name: str, args: Any) -> tuple[str, str]:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            pass
    return name, json.dumps(args, sort_keys=True, ensure_ascii=False)


def last_call(messages: list[dict]) -> tuple[str, str] | None:
    """The most recent tool call in a request's history, canonicalised for repeat detection."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            fn = msg["tool_calls"][-1].get("function", {})
            return canonical_call(fn.get("name", ""), fn.get("arguments"))
    return None


def history_calls(messages: list[dict]) -> list[tuple[str, str]]:
    calls = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            fn = msg["tool_calls"][-1].get("function", {})
            calls.append(canonical_call(fn.get("name", ""), fn.get("arguments")))
    return calls
