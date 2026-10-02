"""g4kit-log-stats: per-session diagnostics from g4kit-proxy full logs (``*_full.jsonl``).

Each request carries the whole session history, so a session is reconstructed from its longest logged request
(sessions are keyed by role and the first user message). Reported per session and per role:

* LLM calls, history tool calls, the longest run of identical consecutive calls and the most repeated call
  (catches A-B-A-B loops);
* edit_file outcomes read from the tool results that follow each edit_file call: ok, not found, other error;
* malformed or unparsed calls in the raw outputs, and requests rejected for context overflow.

Roles default to the start of the system prompt; ``--role NAME=REGEX`` names them.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .toolcalls import canonical_call, classify_calls, declared_tool_names


@dataclass
class Session:
    role: str
    key: str
    requests: int = 0
    longest: list = field(default_factory=list)
    malformed: int = 0
    unparsed: int = 0
    overflow: int = 0

    def calls(self) -> list[tuple[str, str]]:
        out = []
        for m in self.longest:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    out.append(canonical_call(fn.get("name", ""), fn.get("arguments")))
        return out

    def longest_run(self) -> int:
        best = run = 0
        prev = None
        for c in self.calls():
            run = run + 1 if c == prev else 1
            best, prev = max(best, run), c
        return best

    def most_repeated(self) -> int:
        counts = Counter(self.calls())
        return max(counts.values()) if counts else 0

    def edit_outcomes(self) -> Counter:
        ids, out = {}, Counter()
        for m in self.longest:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    if tc.get("function", {}).get("name") == "edit_file":
                        ids[tc.get("id")] = True
            elif m.get("role") == "tool" and m.get("tool_call_id") in ids:
                out[edit_outcome(m.get("content"))] += 1
        return out


def edit_outcome(content) -> str:
    c = content if isinstance(content, str) else json.dumps(content)
    if '"status": "error"' in c or '\\"status\\": \\"error\\"' in c:
        return "not_found" if "not found" in c.lower() else "error"
    return "ok"


def system_text(msgs: list[dict]) -> str:
    if not msgs or msgs[0].get("role") not in ("system", "developer"):
        return ""
    c = msgs[0].get("content")
    return c if isinstance(c, str) else " ".join(p.get("text", "") for p in c or [] if isinstance(p, dict))


def make_role_fn(specs: list[str] | None):
    rules = []
    for spec in specs or []:
        name, sep, pattern = spec.partition("=")
        if not sep:
            raise ValueError(f"--role expects NAME=REGEX, got {spec!r}")
        rules.append((name, re.compile(pattern)))

    def role(msgs: list[dict]) -> str:
        text = system_text(msgs)
        for name, pattern in rules:
            if pattern.search(text):
                return name
        return text.split("\n", 1)[0][:40] or "(no system prompt)"

    return role


def first_user(msgs: list[dict]) -> str:
    for m in msgs:
        if m.get("role") == "user":
            c = m.get("content")
            return (c if isinstance(c, str) else json.dumps(c))[:200]
    return ""


def build_sessions(entries: Iterable[dict], role_fn) -> dict[tuple[str, str], Session]:
    sessions: dict[tuple[str, str], Session] = {}
    for d in entries:
        msgs = (d.get("body") or {}).get("messages") or []
        if not msgs:
            continue
        key = (role_fn(msgs), first_user(msgs))
        s = sessions.setdefault(key, Session(role=key[0], key=key[1]))
        s.requests += 1
        if len(msgs) > len(s.longest):
            s.longest = msgs
        if d.get("status", 200) == 400 and "context" in str(d.get("error") or "").lower():
            s.overflow += 1
        raw = d.get("raw") or ""
        if raw:
            info = classify_calls(raw, declared_tool_names((d.get("body") or {}).get("tools")))
            s.malformed += bool(info["malformed_names"])
            s.unparsed += bool(info["unparsed"])
    return sessions


def report(sessions: dict, loop_threshold: int = 10, top: int = 10) -> str:
    by_role: dict[str, list[Session]] = defaultdict(list)
    for s in sessions.values():
        by_role[s.role].append(s)
    lines = [f"{'role':40s} {'sessions':>8s} {'llm calls':>9s} {'edit ok':>7s} {'not found':>9s} {'edit err':>8s} "
             f"{'loops>=' + str(loop_threshold):>9s} {'malformed':>9s} {'unparsed':>8s} {'overflow':>8s}"]
    for role, ss in sorted(by_role.items()):
        edits = sum((s.edit_outcomes() for s in ss), Counter())
        lines.append(f"{role[:40]:40s} {len(ss):8d} {sum(s.requests for s in ss):9d} {edits['ok']:7d} "
                     f"{edits['not_found']:9d} {edits['error']:8d} "
                     f"{sum(1 for s in ss if s.longest_run() >= loop_threshold):9d} "
                     f"{sum(s.malformed for s in ss):9d} {sum(s.unparsed for s in ss):8d} {sum(s.overflow for s in ss):8d}")
    worst = sorted(sessions.values(), key=lambda s: -s.most_repeated())[:top]
    worst = [s for s in worst if s.most_repeated() >= 2]
    if worst:
        lines.append("\nmost repetitive sessions (longest identical run / most repeated call / llm calls):")
        for s in worst:
            lines.append(f"  {s.role[:30]:30s} run {s.longest_run():3d}  most {s.most_repeated():3d}  calls {s.requests:4d}"
                         f"  {s.key[:70]!r}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="g4kit-log-stats", description=__doc__.split("\n\n")[0])
    ap.add_argument("logs", nargs="+", help="g4kit-proxy *_full.jsonl file(s)")
    ap.add_argument("--role", action="append", metavar="NAME=REGEX", help="name roles by a regex on the system prompt")
    ap.add_argument("--loop-threshold", type=int, default=10, help="identical consecutive calls counted as a loop")
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args(argv)
    entries = []
    for path in a.logs:
        with open(path, encoding="utf-8") as f:
            entries.extend(json.loads(line) for line in f if line.strip())
    print(report(build_sessions(entries, make_role_fn(a.role)), a.loop_threshold, a.top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
