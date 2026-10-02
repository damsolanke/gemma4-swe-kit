"""g4kit-replay: resend logged requests under prompt or sampling changes and measure what the model does next.

Contexts come from g4kit-proxy full logs (``*_full.jsonl``: request body plus the raw output that was logged).
A selector picks the decision points to study; each condition edits the request (system-prompt substitutions,
sampling, thinking) and N continuations are sampled per context and condition through an OpenAI-compatible
endpoint: g4kit-proxy (returns the raw generation as well) or a real vLLM server. Metrics are pluggable.

Selectors:
  all            every logged request
  malformed      the logged output had a malformed tool name; with --onset (default) only contexts whose history
                 holds no earlier unparsed call or harness "token limit" nudge
  after-error    the last tool result in the context is an error
  repeat         the context already ends in two identical calls (loop continuation)
  repeat-onset   the logged output repeated the previous call, and the context itself does not end in a repeat
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics as st
import sys
import urllib.request
from collections import defaultdict
from typing import Iterable

from ..toolcalls import (TOOL_CALL_RE, TOOL_CALL_START, builtin_parse_args, canonical_call, classify_calls,
                         declared_tool_names, extract_reasoning, history_calls, last_call, strip_stop)
from . import conditions as cond_mod
from . import metrics as met

NUDGE_MARKERS = ("token limit before the tool call finished closing",)


def load_contexts(paths: Iterable[str]) -> list[dict]:
    out = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if "body" not in d and "messages" in d:
                    d = {"body": d, "raw": None}
                if d.get("status", 200) != 200 or not (d.get("body") or {}).get("messages"):
                    continue
                out.append(d)
    return out


def _system(msgs: list[dict]) -> str:
    if not msgs or msgs[0].get("role") not in ("system", "developer"):
        return ""
    c = msgs[0].get("content")
    return c if isinstance(c, str) else "".join(p.get("text", "") for p in c or [] if isinstance(p, dict))


def _has_prior_malformation(msgs: list[dict]) -> bool:
    for m in msgs:
        text = m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))
        if m.get("role") == "user" and any(k in (text or "") for k in NUDGE_MARKERS):
            return True
        if m.get("role") == "assistant" and TOOL_CALL_START in (text or ""):
            return True
    return False


def _last_tool_error(msgs: list[dict]) -> bool:
    for m in reversed(msgs):
        if m.get("role") == "tool":
            c = m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))
            return '"status": "error"' in (c or "") or '\\"status\\": \\"error\\"' in (c or "")
    return False


def _first_raw_call(raw: str | None) -> tuple[str, str] | None:
    if not raw:
        return None
    _, content = extract_reasoning(strip_stop(raw))
    m = TOOL_CALL_RE.search(content or "")
    if not m:
        return None
    return canonical_call(m.group(1), builtin_parse_args(m.group(2)))


def select(entries: list[dict], how: str, *, onset: bool = True, role: str | None = None) -> list[dict]:
    role_re = re.compile(role) if role else None
    out = []
    for d in entries:
        body, raw = d["body"], d.get("raw")
        msgs = body["messages"]
        if role_re and not role_re.search(_system(msgs)):
            continue
        calls = history_calls(msgs)
        if how == "all":
            keep = True
        elif how == "malformed":
            keep = raw is not None and bool(classify_calls(raw, declared_tool_names(body.get("tools")))["malformed_names"])
            keep = keep and not (onset and _has_prior_malformation(msgs))
        elif how == "after-error":
            keep = _last_tool_error(msgs)
        elif how == "repeat":
            keep = len(calls) >= 2 and calls[-1] == calls[-2]
        elif how == "repeat-onset":
            nxt = _first_raw_call(raw)
            keep = bool(calls) and nxt == calls[-1] and not (len(calls) >= 2 and calls[-1] == calls[-2])
        else:
            raise ValueError(f"unknown selector {how!r}")
        if keep:
            out.append(d)
    return out


def dedupe_sessions(entries: list[dict]) -> list[dict]:
    """Keep the first context of each session (role + first user message)."""
    seen, out = set(), []
    for d in entries:
        msgs = d["body"]["messages"]
        user = next((m.get("content") for m in msgs if m.get("role") == "user"), "")
        key = (_system(msgs)[:200], str(user)[:300])
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def post(api_base: str, body: dict, api_key: str = "local", timeout: float = 3600.0) -> dict:
    req = urllib.request.Request(api_base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def sign_test(better: int, worse: int) -> float:
    """Two-sided exact sign test p-value."""
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def summarize(records: list[dict], metric_names: list[str], cond_names: list[str]) -> str:
    lines = []
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cond[r["cond"]].append(r)
    header = f"{'condition':14s} {'n':>5s} " + " ".join(f"{m[:16]:>16s}" for m in metric_names)
    lines.append(header)
    for c in cond_names:
        rs = by_cond.get(c, [])
        cells = []
        for m in metric_names:
            vals = [r["metrics"][m] for r in rs if r["metrics"].get(m) is not None]
            if not vals:
                cells.append(f"{'-':>16s}")
            elif all(isinstance(v, bool) for v in vals):
                cells.append(f"{sum(vals):>6d} ({100 * sum(vals) / len(vals):5.1f}%)")
            else:
                cells.append(f"{st.mean(vals):16.1f}")
        lines.append(f"{c:14s} {len(rs):5d} " + " ".join(cells))
    if len(cond_names) >= 2:
        base = cond_names[0]
        for other in cond_names[1:]:
            for m in metric_names:
                per_ctx: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
                for r in records:
                    if r["cond"] in (base, other) and r["metrics"].get(m) is not None:
                        per_ctx[r["ctx"]][r["cond"]].append(float(r["metrics"][m]))
                lower = higher = tied = 0
                for v in per_ctx.values():
                    if not v[base] or not v[other]:
                        continue
                    d = st.mean(v[other]) - st.mean(v[base])
                    lower += d < 0
                    higher += d > 0
                    tied += d == 0
                if lower + higher:
                    lines.append(f"paired {other} vs {base}, {m}: lower in {lower} contexts, higher in {higher}, "
                                 f"tied {tied}; sign test p = {sign_test(lower, higher):.3g}")
    return "\n".join(lines)


def run(contexts: list[dict], conds: list[cond_mod.Condition], metric_fns: dict, *, api_base: str, n: int,
        out_path: str | None, model: str | None = None, require_subs: bool = False, api_key: str = "local") -> list[dict]:
    records: list[dict] = []
    fout = open(out_path, "w", encoding="utf-8") if out_path else None
    try:
        for i, d in enumerate(contexts):
            prev = last_call(d["body"]["messages"])
            edited = [(c, *c.apply(d["body"])) for c in conds]
            if require_subs and any(c.edits_system and k == 0 for c, _, k in edited):
                continue
            for c, body, n_subs in edited:
                body["g4kit_return_raw"] = True
                if model:
                    body["model"] = model
                for k in range(n):
                    resp = post(api_base, body, api_key)
                    choice = resp["choices"][0]
                    sample = met.Sample(body=body, message=choice["message"], finish=choice.get("finish_reason") or "",
                                        raw=(resp.get("g4kit") or {}).get("raw"), usage=resp.get("usage") or {},
                                        prev_call=prev)
                    vals = {name: fn(sample) for name, fn in metric_fns.items()}
                    rec = {"ctx": i, "cond": c.name, "k": k, "subs": n_subs, "metrics": vals,
                           "finish": sample.finish, "raw": (sample.raw if sample.raw is not None
                                                            else (sample.message.get("content") or ""))[:2000]}
                    records.append(rec)
                    if fout:
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fout.flush()
            print(f"context {i + 1}/{len(contexts)} done", flush=True)
    finally:
        if fout:
            fout.close()
    return records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="g4kit-replay", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=cond_mod.__doc__)
    ap.add_argument("logs", nargs="+", help="g4kit-proxy *_full.jsonl file(s), or JSONL of request bodies")
    ap.add_argument("--select", default="all", choices=["all", "malformed", "after-error", "repeat", "repeat-onset"])
    ap.add_argument("--no-onset", action="store_true", help="with --select malformed: keep later malformations too")
    ap.add_argument("--role", help="regex on the system prompt")
    ap.add_argument("--dedupe", action="store_true", help="one context per agent session")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="shuffle seed when --limit is used")
    ap.add_argument("--cond", action="append", default=[], help="NAME=op:arg,... (repeatable; the first is the baseline)")
    ap.add_argument("--require-subs", action="store_true",
                    help="skip contexts where a system-prompt substitution condition changes nothing")
    ap.add_argument("--metric", action="append", help="metric name or module:function (repeatable); "
                                                      f"default: {', '.join(met.DEFAULT)}")
    ap.add_argument("--n", type=int, default=4, help="samples per context and condition")
    ap.add_argument("--api-base", default="http://127.0.0.1:11436/v1")
    ap.add_argument("--api-key", default="local")
    ap.add_argument("--model", help="override the model field (e.g. a LoRA adapter name)")
    ap.add_argument("--out", default="replay.jsonl")
    ap.add_argument("--dry", action="store_true", help="only count the selected contexts")
    a = ap.parse_args(argv)
    contexts = select(load_contexts(a.logs), a.select, onset=not a.no_onset, role=a.role)
    if a.dedupe:
        contexts = dedupe_sessions(contexts)
    if a.limit and len(contexts) > a.limit:
        random.Random(a.seed).shuffle(contexts)
        contexts = contexts[: a.limit]
    print(f"{len(contexts)} contexts selected ({a.select})", flush=True)
    if a.dry or not contexts:
        return 0
    conds = [cond_mod.parse(s) for s in (a.cond or ["orig="])]
    metric_fns = met.resolve(a.metric)
    records = run(contexts, conds, metric_fns, api_base=a.api_base, n=a.n, out_path=a.out, model=a.model,
                  require_subs=a.require_subs, api_key=a.api_key)
    print(summarize(records, list(metric_fns), [c.name for c in conds]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
