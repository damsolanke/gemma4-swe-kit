"""g4kit-scorer-time: estimate how long a local run would take on the scorer (4x L4, tasks run one at a time).

Local model time is replaced by a fit to scorer-stack traces: each LLM call costs PER_CALL seconds plus
completion_tokens / TOK_S (0.6 s + tokens / 25.5 tok/s, measured on 4x L4 with three evaluation arms sharing one vLLM
server, so a single-stream scorer is somewhat faster and the estimate is conservative). Prompt length adds
nothing measurable because of prefix caching. Tool and sandbox time is kept as measured locally
(task wall time minus local model time), and the per-task estimate is capped at the submission's time limit.

The projection for the hidden set multiplies the mean per-task estimate plus a setup allowance (about one
minute per task for sandbox setup on the scorer) by the task count, to compare with the 12-hour limit.

Inputs: the results JSONL written by ``g4kit-harness run`` (needs ``ended_at`` and ``wall_s`` per task,
``started_at`` when present) and the proxy's summary log (``t``, ``dt``, ``usage.completion_tokens``).
Use one proxy log per run, or runs that did not overlap in time.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PER_CALL_S = 0.6
TOK_S = 25.5
LIMIT_HOURS = 12.0


@dataclass
class TaskEstimate:
    task: str
    resolved: bool
    has_patch: bool
    wall_s: float
    n_calls: int
    n_thinking: int
    completion_tokens: int
    llm_local_s: float
    llm_est_s: float
    tool_s: float
    est_s: float


def load_jsonl(paths: Iterable[str | Path]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def task_window(rec: dict, slack: float = 2.0) -> tuple[float, float]:
    end = float(rec["ended_at"])
    start = float(rec["started_at"]) if rec.get("started_at") is not None else end - float(rec["wall_s"])
    return start - slack, end + slack


def estimate_task(rec: dict, calls: list[dict], *, per_call: float = PER_CALL_S, tok_s: float = TOK_S,
                  cap_s: float | None = None) -> TaskEstimate:
    ok_calls = [c for c in calls if c.get("status", 200) == 200]
    completion = [int(((c.get("usage") or {}).get("completion_tokens")) or 0) for c in ok_calls]
    llm_local = sum(float(c.get("dt") or 0) for c in ok_calls)
    llm_est = sum(per_call + n / tok_s for n in completion)
    wall = float(rec.get("wall_s") or 0)
    tool = max(0.0, wall - llm_local)
    est = llm_est + tool
    if cap_s:
        est = min(est, cap_s)
    return TaskEstimate(task=str(rec.get("id") or rec.get("task") or "?"), resolved=bool(rec.get("resolved")),
                        has_patch=bool(rec.get("patch_chars")), wall_s=wall, n_calls=len(ok_calls),
                        n_thinking=sum(1 for c in ok_calls if c.get("enable_thinking")),
                        completion_tokens=sum(completion), llm_local_s=llm_local, llm_est_s=llm_est,
                        tool_s=tool, est_s=est)


def estimate_run(results: list[dict], calls: list[dict], **kw) -> list[TaskEstimate]:
    calls = sorted(calls, key=lambda c: float(c.get("t") or 0))
    out = []
    for rec in results:
        lo, hi = task_window(rec)
        mine = [c for c in calls if lo <= float(c.get("t") or 0) <= hi]
        out.append(estimate_task(rec, mine, **kw))
    return out


def project_hours(estimates: list[TaskEstimate], n_tasks: int = 120, setup_min: float = 1.0) -> float:
    if not estimates:
        return 0.0
    return (st.mean(e.est_s for e in estimates) + setup_min * 60) * n_tasks / 3600


def report(estimates: list[TaskEstimate], *, n_tasks: int, setup_min: float, title: str = "") -> str:
    lines = [f"== {title}: {len(estimates)} tasks" if title else f"== {len(estimates)} tasks"]
    for e in estimates:
        mark = "S" if e.resolved else ("p" if e.has_patch else "-")
        lines.append(f"  {e.task:24s} {mark} local {e.wall_s:6.0f}s  llm_calls {e.n_calls:3d} (thinking {e.n_thinking:3d})"
                     f"  completion {e.completion_tokens:6d} tok -> est scorer {e.est_s / 60:5.1f} min"
                     f" (llm {e.llm_est_s / 60:4.1f} + tools {e.tool_s / 60:4.1f})")
    if estimates:
        hours = project_hours(estimates, n_tasks, setup_min)
        flag = "OVER the 12 h limit" if hours > LIMIT_HOURS else ("within 1.5 h of the limit" if hours > LIMIT_HOURS - 1.5 else "fits")
        lines.append(f"  mean est {st.mean(e.est_s for e in estimates) / 60:.2f} min/task, median "
                     f"{st.median(e.est_s for e in estimates) / 60:.2f}; x{n_tasks} tasks + {setup_min:g} min setup each = "
                     f"{hours:.1f} h ({flag}); solved {sum(e.resolved for e in estimates)}/{len(estimates)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="g4kit-scorer-time", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True, help="results JSONL file(s) from g4kit-harness run")
    ap.add_argument("--proxy-log", nargs="+", required=True, help="g4kit-proxy summary log(s) covering the run")
    ap.add_argument("--per-call", type=float, default=PER_CALL_S, help="fixed seconds per LLM call on the scorer")
    ap.add_argument("--tok-s", type=float, default=TOK_S, help="scorer decode speed, tokens/s")
    ap.add_argument("--cap-minutes", type=float, default=None, help="the submission's max_time_minutes")
    ap.add_argument("--tasks", type=int, default=120, help="hidden task count for the projection")
    ap.add_argument("--setup-minutes", type=float, default=1.0, help="per-task setup allowance on the scorer")
    ap.add_argument("--by-arm", action="store_true", help="report each arm (results 'arm' field) separately")
    ap.add_argument("--json", action="store_true", help="print per-task estimates as JSON lines")
    a = ap.parse_args(argv)
    results, calls = load_jsonl(a.results), load_jsonl(a.proxy_log)
    groups: dict[str, list[dict]] = {}
    for rec in results:
        groups.setdefault(str(rec.get("arm", "")) if a.by_arm else "", []).append(rec)
    for name, recs in groups.items():
        ests = estimate_run(recs, calls, per_call=a.per_call, tok_s=a.tok_s,
                            cap_s=a.cap_minutes * 60 if a.cap_minutes else None)
        if a.json:
            for e in ests:
                print(json.dumps(dict(e.__dict__, arm=name)))
        else:
            print(report(ests, n_tasks=a.tasks, setup_min=a.setup_minutes, title=name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
