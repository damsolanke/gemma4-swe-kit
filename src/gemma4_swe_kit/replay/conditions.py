"""Replay conditions: named, comma-separated edits applied to a logged request before it is resent.

  NAME=                      the request as logged
  NAME=subs:FILE             regex substitutions on the system prompt, FILE = JSON [[pattern, replacement], ...]
  NAME=append:FILE           append FILE's text to the system prompt (prepend:FILE likewise)
  NAME=temp:0.6              sampling overrides: temp, top_p, top_k, seed, max_tokens
  NAME=think:on              chat_template_kwargs.enable_thinking on/off
  NAME=budget:256            thinking_token_budget

Example: --cond orig= --cond prose=subs:examples/prose_rewrites.json --cond hot=temp:0.6
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Condition:
    name: str
    ops: list[tuple[str, str]]

    def apply(self, body: dict) -> tuple[dict, int]:
        """(edited copy of the body, number of system-prompt substitutions made)."""
        b = copy.deepcopy(body)
        n_subs = 0
        for op, arg in self.ops:
            if op in ("subs", "append", "prepend"):
                msgs = b.get("messages") or []
                if not msgs or msgs[0].get("role") not in ("system", "developer"):
                    continue
                text = _system_text(msgs[0])
                if op == "subs":
                    for pattern, repl in _load_subs(arg):
                        text, k = re.subn(pattern, repl, text)
                        n_subs += k
                elif op == "append":
                    text = text + Path(arg).read_text(encoding="utf-8")
                else:
                    text = Path(arg).read_text(encoding="utf-8") + text
                msgs[0]["content"] = text
            elif op == "temp":
                b["temperature"] = float(arg)
            elif op == "top_p":
                b["top_p"] = float(arg)
            elif op in ("top_k", "seed"):
                b[op] = int(arg)
            elif op == "max_tokens":
                b.pop("max_completion_tokens", None)
                b["max_tokens"] = int(arg)
            elif op == "think":
                kw = dict(b.get("chat_template_kwargs") or {})
                kw["enable_thinking"] = arg.lower() in ("on", "true", "1", "yes")
                b["chat_template_kwargs"] = kw
            elif op == "budget":
                b["thinking_token_budget"] = int(arg)
            else:
                raise ValueError(f"unknown condition op {op!r}")
        return b, n_subs

    @property
    def edits_system(self) -> bool:
        return any(op in ("subs", "append", "prepend") for op, _ in self.ops)


_SUBS_CACHE: dict[str, list[tuple[str, str]]] = {}


def _load_subs(path: str) -> list[tuple[str, str]]:
    if path not in _SUBS_CACHE:
        pairs = json.loads(Path(path).read_text(encoding="utf-8"))
        _SUBS_CACHE[path] = [(p[0], p[1]) for p in pairs]
    return _SUBS_CACHE[path]


def _system_text(msg: dict) -> str:
    c = msg.get("content")
    return c if isinstance(c, str) else "".join(p.get("text", "") for p in c or [] if isinstance(p, dict))


def parse(spec: str) -> Condition:
    name, sep, rest = spec.partition("=")
    if not sep or not name:
        raise ValueError(f"condition must look like NAME=op:arg,...; got {spec!r}")
    ops = []
    for item in filter(None, (x.strip() for x in rest.split(","))):
        op, colon, arg = item.partition(":")
        if not colon:
            raise ValueError(f"condition op must look like op:arg, got {item!r}")
        ops.append((op, arg))
    return Condition(name, ops)
