"""g4kit-curate: score converted teacher trajectories (g4kit-convert-openhands output) and keep the best ~40% for
distillation.

Input lines are {instance_id, repo, issue, n_turns, messages} (OpenAI-style; tool contents are the harness's result
JSON). Every trajectory in the input is resolved (the converter keeps resolved runs only, at most 2 per instance).
The output keeps the input format, so g4kit-render-distill renders it unchanged.

Normalization (before scoring; --no-normalize turns it off)
  The converter maps the teacher's first two directory views (/workspace, then /workspace/<repo>) to the same
  'find . -maxdepth 2' command, so about half of the trajectories open with two identical calls whose outputs differ
  (the first is the shallower listing). That teaches 'run the same command twice'. For two adjacent single-call turns
  with identical tool name and arguments: if both results are identical the later turn is dropped (a pure repeat);
  if every line of the earlier result appears in the later one the earlier turn is dropped (it is superseded). The
  dropped turn's reasoning moves to the kept turn, so no teacher note is lost.

Signals (each in [0, 1], 1 = best), weights in WEIGHTS:
  order     inspect -> edit -> verify. 0.4 x share of edited existing files that were inspected (read_file of the
            path, or a shell command naming it) before their first edit + 0.4 x a test or script run after the last
            repository edit + 0.2 x a run before the first edit (reproduction).
  repeat    exp(-0.5 x redundant), redundant = calls identical (name + arguments) to an earlier call with no
            repository edit in between (re-running a check after an edit is not redundant).
  errors    1 - 8 x tool-error rate, floored at 0. Tool errors follow SWE-Lego's error masking: file-tool errors
            (read_file, edit_file, write_file) and shell invocation errors (command not found, missing path, usage
            errors, pytest exit 4/5); a failing test or reproduction run is NOT an error. Forbidden actions in the
            harness (pip/conda install, curl/wget: the scorer is offline) count as three errors each.
  retry     exp(-0.7 x failed retries), a failed retry = a tool error directly followed by another tool error from
            the same tool.
  patch     small final patch (the teacher's model_patch from the source trajectories, --source). Size counts changed
            lines in source .py files only (not tests, not files left at the repository root, not build metadata): 1
            up to 15 lines, then 1 - log2(changed / 15) / 4 (0 at 240); x 1 / 0.85 / 0.7 / 0.5 for <= 2 / 3 / 4 / 5+
            source files; x 0.3 if it modifies an existing test file, x 0.6 if it leaves a new file at the
            repository root or build metadata (*.egg-info) behind, x 0.8 if it adds a test file, x 0.8 if it touches
            packaging/config files. Edited tests make the hidden test_patch fail to apply on the scorer, and scratch
            files belong in /tmp, so the deployed agent must not learn either.
  brevity   exp(-0.5 x long thoughts), a long thought = a non-first turn whose reasoning exceeds 2,000 characters
            (~500 tokens, the deployment thinking budget); the first turn's plan is exempt.

Selection (deterministic, no randomness): target = round(keep x eligible). Greedy by effective score = score -
repo_penalty x (trajectories already kept from that repository) - instance_penalty if the other trajectory of the
same instance is already kept; at most --max-per-repo per repository and 2 per instance; ties broken by instance id
and input position. Repositories in --exclude-repos are not eligible, nor are trajectories whose final patch is not
found in --source.

Outputs: OUT.jsonl (kept trajectories, normalized, input order), OUT.summary.json (counts, score distribution, feature
means, repositories covered), OUT.scores.jsonl (every input trajectory: features, score, kept flag).
"""
from __future__ import annotations

import argparse
import collections
import heapq
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

WEIGHTS = {"order": 0.25, "repeat": 0.20, "errors": 0.20, "retry": 0.10, "patch": 0.20, "brevity": 0.05}
LONG_THOUGHT_CHARS = 2000
PATCH_COLUMNS = ["instance_id", "resolved", "model_patch"]

RUN_PROGS = {"python", "pytest", "py.test", "nosetests", "tox", "nox", "coverage", "trial", "unittest"}
INSPECT_PROGS = {"grep", "egrep", "fgrep", "rg", "ag", "find", "ls", "cat", "head", "tail", "sed", "awk", "wc", "tree",
                 "nl", "less", "more", "file", "stat", "git", "pwd", "which", "type", "du"}
INVOCATION_ERR = re.compile(
    r"command not found|No such file or directory|can't open file|cannot access|Permission denied|Is a directory|"
    r"unrecognized arguments|invalid option|unknown option|illegal option|usage:|Usage:|syntax error near|"
    r"ERROR: file or directory not found|ERROR: not found:|no tests ran|ERROR: usage")
FORBIDDEN = re.compile(r"\bpip3?\s+install\b|\bpython3? -m pip install\b|\bconda install\b|\bcurl\b|\bwget\b")
TEST_PATH = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")
CONFIG_PATH = re.compile(r"(^|/)(pyproject\.toml|setup\.cfg|setup\.py|tox\.ini|pytest\.ini|noxfile\.py|"
                         r"requirements[^/]*\.txt|MANIFEST\.in|\.pre-commit-config\.yaml)$")
SHELL_EDIT = re.compile(r"\bsed\s+-i|\bperl\s+-[a-z]*i|\bgit\s+apply\b|\bpatch\s+-p")


# ------------------------------------------------------------------------------------------------ parsing
def calls_of(conv: dict) -> tuple[list[tuple[int, str, str, dict]], dict[str, dict]]:
    """[(turn_idx, call_id, name, args)] in order, and {call_id: result dict}."""
    calls, res = [], {}
    for k, m in enumerate(conv["messages"]):
        if m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                calls.append((k, tc["id"], tc["function"]["name"], tc["function"]["arguments"]))
        elif m["role"] == "tool":
            res[m["tool_call_id"]] = json.loads(m["content"])
    return calls, res


def main_program(cmd: str) -> tuple[str, str]:
    """First program of a shell command after 'cd X &&', env assignments and 'timeout N'."""
    s = cmd.strip()
    while True:
        m = re.match(r"^cd\s+\S+\s*(&&|;)\s*", s)
        if not m:
            break
        s = s[m.end():]
    toks = s.split()
    while toks and (re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", toks[0]) or toks[0] in ("export", "env", "time")):
        toks = toks[1:]
    if toks and toks[0] == "timeout":
        toks = toks[2:]
    if not toks:
        return "", ""
    prog = os.path.basename(toks[0])
    if re.match(r"^python[0-9.]*$", prog):
        prog = "python"
    return prog, " ".join(toks[1:])


def classify_command(cmd: str) -> str:
    """'run' (tests or a script), 'inspect', 'edit' (shell write to a repository file), 'scratch' (write under /tmp)
    or 'other'. Only the first line matters (heredoc bodies follow it)."""
    first = cmd.strip().split("\n", 1)[0]
    prog, rest = main_program(first)
    if SHELL_EDIT.search(first):
        return "scratch" if "/tmp/" in first else "edit"
    if prog in ("cat", "tee", "echo", "printf") and re.search(r">\s*\S", rest):
        return "scratch" if "/tmp/" in rest else "edit"
    if prog in RUN_PROGS or (prog.endswith(".sh") and "test" in prog) or (prog == "make" and "test" in rest):
        return "run"
    if prog in INSPECT_PROGS:
        return "inspect"
    return "other"


def is_test_command(cmd: str) -> bool:
    prog, rest = main_program(cmd)
    return prog in ("pytest", "py.test", "nosetests", "tox", "nox") or "-m pytest" in cmd or "-m unittest" in cmd \
        or (prog == "make" and "test" in rest)


def tool_error(name: str, args: dict, result: dict) -> bool:
    """True for errors the teacher caused by a wrong call (masked by SWE-Lego), False for expected failures."""
    if result.get("status") != "error":
        return False
    if name in ("read_file", "edit_file", "write_file"):
        return True
    if name != "run_command":
        return True
    cmd = args.get("command", "")
    det = result.get("details") or {}
    code = det.get("exit_code")
    out = (result.get("error_message") or "") + "\n" + (det.get("stderr") or "")
    if code in (126, 127):
        return True
    if is_test_command(cmd):
        return code in (4, 5)                     # pytest usage error / nothing collected; failures are expected
    prog, _ = main_program(cmd)
    if prog == "python":
        return bool(re.search(r"can't open file|No such file or directory: '/", out[:400])) and "Traceback" not in out
    if prog in INSPECT_PROGS or prog in ("cd", "rm", "mv", "cp", "mkdir"):
        if code == 1 and prog in ("grep", "egrep", "fgrep", "rg", "ag") and not out.strip():
            return False                          # grep found nothing: a valid answer, not an error
        return bool(INVOCATION_ERR.search(out)) or code not in (0, 1)
    return bool(INVOCATION_ERR.search(out[:2000]))


# ------------------------------------------------------------------------------------------------ normalization
def normalize(conv: dict) -> tuple[dict, int]:
    """Collapse adjacent identical single-call turns (see the module docstring). Returns (conv, n_dropped)."""
    msgs = conv["messages"]
    turns: list[list] = []                        # [assistant msg, [tool msgs]] in order; final text kept apart
    for m in msgs:
        if m["role"] == "assistant" and m.get("tool_calls"):
            turns.append([dict(m), []])
        elif m["role"] == "tool":
            turns[-1][1].append(m)
    tail = [m for m in msgs if m["role"] == "assistant" and not m.get("tool_calls")]
    out: list[list] = []
    dropped = 0
    for t in turns:
        if out and len(t[0]["tool_calls"]) == 1 and len(out[-1][0]["tool_calls"]) == 1 and len(t[1]) == 1 \
                and len(out[-1][1]) == 1:
            p, c = out[-1][0]["tool_calls"][0]["function"], t[0]["tool_calls"][0]["function"]
            if p["name"] == c["name"] and p["arguments"] == c["arguments"] and c["name"] != "submit_patch":
                r_prev, r_cur = out[-1][1][0]["content"], t[1][0]["content"]
                reason = "\n\n".join(x for x in (out[-1][0].get("reasoning"), t[0].get("reasoning")) if x) or None
                if r_prev == r_cur:               # pure repeat: drop the later turn
                    out[-1][0]["reasoning"] = reason
                    dropped += 1
                    continue
                prev_lines = set(_result_lines(r_prev))
                if prev_lines and prev_lines <= set(_result_lines(r_cur)):   # superseded: drop the earlier turn
                    t[0]["reasoning"] = reason
                    out[-1] = t
                    dropped += 1
                    continue
        out.append(t)
    if not dropped:
        return conv, 0
    new = []
    for a, tools in out:
        new.append(a)
        new.extend(tools)
    new.extend(tail)
    conv = dict(conv)
    conv["messages"] = new
    conv["n_turns"] = len(out)
    return conv, dropped


def _result_lines(content: str) -> list[str]:
    d = json.loads(content)
    text = d.get("stdout") or d.get("content") or ""
    return [line for line in text.split("\n") if line.strip()]


# ------------------------------------------------------------------------------------------------ patches
def iter_patch_rows(src: str) -> Iterator[tuple[str, Any, str | None]]:
    """(instance_id, resolved, model_patch) in the order g4kit-convert-openhands reads the same source: a parquet
    file, a directory of parquet files (sorted by path) or a JSONL file."""
    path = Path(src)
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    yield row["instance_id"], row.get("resolved"), row.get("model_patch")
        return
    import pyarrow.parquet as pq

    for file in (sorted(path.glob("**/*.parquet")) if path.is_dir() else [path]):
        t = pq.ParquetFile(file).read(columns=PATCH_COLUMNS)
        yield from zip(t.column("instance_id").to_pylist(), t.column("resolved").to_pylist(),
                       t.column("model_patch").to_pylist())


def load_patches(convs: list[dict], rows: Iterable[tuple[str, Any, str | None]]) -> list[str | None]:
    """Teacher model_patch for each conversation: the converter wrote trajectories in source row order, so for each
    instance the k-th converted line is the next resolved row whose patch length equals submit_patch's patch_size."""
    by_instance: dict[str, list[str]] = collections.defaultdict(list)
    for iid, ok, p in rows:
        if ok:
            by_instance[iid].append(p or "")
    ptr: collections.Counter = collections.Counter()
    out: list[str | None] = []
    for conv in convs:
        sub = [m for m in conv["messages"] if m["role"] == "tool" and m.get("name") == "submit_patch"]
        d = json.loads(sub[-1]["content"]) if sub else {}
        size, nfiles = d.get("patch_size"), d.get("files_changed")
        lst, p = by_instance.get(conv["instance_id"], []), ptr[conv["instance_id"]]
        while p < len(lst) and not (len(lst[p]) == size and lst[p].count("diff --git") == nfiles):
            p += 1
        if p < len(lst):
            out.append(lst[p])
            ptr[conv["instance_id"]] = p + 1
        else:
            out.append(None)
    return out


def patch_stats(patch: str) -> dict:
    files: list[dict] = []
    cur = None
    for line in patch.split("\n"):
        if line.startswith("diff --git "):
            m = re.match(r"diff --git a/(\S+) b/(\S+)", line)
            cur = {"path": m.group(2) if m else line.split()[-1][2:], "new": False, "deleted": False, "add": 0, "rem": 0}
            files.append(cur)
        elif cur is None:
            continue
        elif line.startswith("new file mode"):
            cur["new"] = True
        elif line.startswith("deleted file mode"):
            cur["deleted"] = True
        elif line.startswith("+") and not line.startswith("+++"):
            cur["add"] += 1
        elif line.startswith("-") and not line.startswith("---"):
            cur["rem"] += 1
    tests = [f for f in files if TEST_PATH.search(f["path"])]
    junk = [f for f in files if f not in tests and ((f["new"] and "/" not in f["path"]) or ".egg-info/" in f["path"]
                                                     or "/EGG-INFO/" in f["path"])]
    src = [f for f in files if f not in tests and f not in junk and f["path"].endswith(".py")]
    return {"files": len(files), "changed": sum(f["add"] + f["rem"] for f in files),
            "src_changed": sum(f["add"] + f["rem"] for f in src), "src_files": len(src),
            "edits_existing_test": any(not f["new"] for f in tests), "adds_test": any(f["new"] for f in tests),
            "adds_root_file": bool(junk), "touches_config": any(CONFIG_PATH.search(f["path"]) for f in files)}


# ------------------------------------------------------------------------------------------------ features
def features(conv: dict, patch: str | None) -> tuple[float, dict, dict]:
    """(score, {signal: value}, raw counts) for one trajectory; patch None scores 0 on the patch signal."""
    calls, res = calls_of(conv)
    n = len(calls)
    kinds, edit_idx, run_idx = [], [], []
    for i, (_, cid, name, args) in enumerate(calls):
        if name in ("edit_file", "write_file"):
            kind = "edit"
        elif name == "read_file":
            kind = "inspect"
        elif name == "run_command":
            kind = classify_command(args.get("command", ""))
        else:
            kind = name
        kinds.append(kind)
        if kind == "edit":
            edit_idx.append(i)
        elif kind == "run":
            run_idx.append(i)
    submit = next((i for i, (_, _, nm, _) in enumerate(calls) if nm == "submit_patch"), n)
    # order
    edited: dict = {}
    for i, (_, _, name, args) in enumerate(calls):
        if name == "edit_file" and args.get("filepath") not in edited:
            edited[args.get("filepath")] = i
    read_ok = 0
    for path, first in edited.items():
        base = os.path.basename(path or "")
        for j in range(first):
            nm, a = calls[j][2], calls[j][3]
            if (nm == "read_file" and a.get("filepath") == path) or \
                    (nm == "run_command" and kinds[j] == "inspect" and base and base in a.get("command", "")):
                read_ok += 1
                break
    read_before_edit = read_ok / len(edited) if edited else 1.0
    last_edit = edit_idx[-1] if edit_idx else -1
    verify_after = any(last_edit < i < submit for i in run_idx) if edit_idx else False
    run_before = any(i < edit_idx[0] for i in run_idx) if edit_idx else False
    f_order = 0.4 * read_before_edit + 0.4 * verify_after + 0.2 * run_before
    # repeats (identical call with no repository edit in between)
    seen: set = set()
    redundant = 0
    for i, (_, _, name, args) in enumerate(calls):
        if kinds[i] == "edit":
            seen = set()
            continue
        key = (name, json.dumps(args, sort_keys=True))
        if key in seen:
            redundant += 1
        seen.add(key)
    # errors
    errs = [tool_error(name, args, res.get(cid, {})) for _, cid, name, args in calls]
    forbidden = sum(1 for _, _, name, args in calls
                    if name == "run_command" and FORBIDDEN.search(args.get("command", "")))
    n_err = sum(errs)
    rate = (n_err + 3 * forbidden) / max(n, 1)
    failed_retries = sum(1 for i in range(1, n) if errs[i] and errs[i - 1] and calls[i][2] == calls[i - 1][2])
    # patch
    ps = patch_stats(patch) if patch is not None else None
    if ps:
        size = 1.0 if ps["src_changed"] <= 15 else max(0.0, 1 - math.log2(ps["src_changed"] / 15) / 4)
        nf = ps["src_files"]
        files_f = 1.0 if nf <= 2 else 0.85 if nf == 3 else 0.7 if nf == 4 else 0.5
        hyg = (0.3 if ps["edits_existing_test"] else 1) * (0.6 if ps["adds_root_file"] else 1) * \
              (0.8 if ps["adds_test"] else 1) * (0.8 if ps["touches_config"] else 1)
        f_patch = size * files_f * hyg
    else:
        f_patch = 0.0
    # brevity
    asst = [m for m in conv["messages"] if m["role"] == "assistant" and m.get("tool_calls")]
    long_thoughts = sum(1 for m in asst[1:] if len(m.get("reasoning") or "") > LONG_THOUGHT_CHARS)
    f = {"order": f_order, "repeat": math.exp(-0.5 * redundant), "errors": max(0.0, 1 - 8 * rate),
         "retry": math.exp(-0.7 * failed_retries), "patch": f_patch, "brevity": math.exp(-0.5 * long_thoughts)}
    score = sum(WEIGHTS[k] * v for k, v in f.items())
    raw = {"calls": n, "turns": conv["n_turns"], "read_before_edit": round(read_before_edit, 3),
           "verify_after_edit": verify_after, "run_before_edit": run_before, "redundant": redundant,
           "tool_errors": n_err, "forbidden": forbidden, "failed_retries": failed_retries,
           "long_thoughts": long_thoughts, "patch": ps}
    return score, {k: round(v, 4) for k, v in f.items()}, raw


# ------------------------------------------------------------------------------------------------ selection
def select(items: list[tuple[int, str, str, float]], target: int, max_per_repo: int, repo_pen: float,
           inst_pen: float) -> list[int]:
    """items: [(idx, instance_id, repo, score)]. Greedy with lazily refreshed effective scores."""
    per_repo: collections.Counter = collections.Counter()
    per_inst: collections.Counter = collections.Counter()
    kept: list[int] = []

    def eff(it: tuple[int, str, str, float]) -> float:
        return it[3] - repo_pen * per_repo[it[2]] - (inst_pen if per_inst[it[1]] else 0)

    heap = [(-it[3], it[1], it[0], it) for it in items]
    heapq.heapify(heap)
    while heap and len(kept) < target:
        neg, iid, idx, it = heapq.heappop(heap)
        if per_repo[it[2]] >= max_per_repo or per_inst[it[1]] >= 2:
            continue
        e = eff(it)
        if e < -neg - 1e-12:
            heapq.heappush(heap, (-e, iid, idx, it))
            continue
        kept.append(idx)
        per_repo[it[2]] += 1
        per_inst[it[1]] += 1
    return kept


def quantiles(xs: list[float], ps: tuple[int, ...] = (0, 10, 25, 50, 75, 90, 100)) -> dict:
    xs = sorted(xs)
    return {f"p{p}": round(xs[min(len(xs) - 1, int(round((len(xs) - 1) * p / 100)))], 4) for p in ps} if xs else {}


def _mean(rows: list[dict], f) -> float | None:
    return round(statistics.mean(f(r) for r in rows), 4) if rows else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="g4kit-curate", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("conv", help="JSONL from g4kit-convert-openhands")
    ap.add_argument("out", help="output JSONL; OUT.summary.json and OUT.scores.jsonl are written next to it")
    ap.add_argument("--source", required=True,
                    help="the trajectories the converter read (parquet file or directory, or .jsonl); the teacher's "
                         "final patches come from their model_patch column")
    ap.add_argument("--keep", type=float, default=0.40, help="share of the eligible trajectories to keep")
    ap.add_argument("--max-per-repo", type=int, default=8)
    ap.add_argument("--repo-penalty", type=float, default=0.02,
                    help="score penalty per trajectory already kept from the same repository")
    ap.add_argument("--instance-penalty", type=float, default=0.03,
                    help="score penalty for an instance whose other trajectory is already kept")
    ap.add_argument("--exclude-repos", default="encode/httpx",
                    help="comma list of repositories that are never kept, e.g. those of your local eval tasks; the "
                         "default is the only repository of the competition's public tasks that occurs in "
                         "nebius/SWE-rebench-openhands-trajectories")
    ap.add_argument("--no-normalize", action="store_true", help="keep adjacent identical calls (see the docstring)")
    a = ap.parse_args(argv)
    excl = set(filter(None, a.exclude_repos.split(",")))
    convs, dropped = [], []
    with open(a.conv, encoding="utf-8") as fin:
        for line in fin:
            c = json.loads(line)
            d = 0
            if not a.no_normalize:
                c, d = normalize(c)
            convs.append(c)
            dropped.append(d)
    patches = load_patches(convs, iter_patch_rows(a.source))
    rows = []
    for i, (c, p) in enumerate(zip(convs, patches)):
        s, f, raw = features(c, p)
        rows.append({"idx": i, "instance_id": c["instance_id"], "repo": c["repo"], "score": round(s, 5),
                     "features": f, "raw": raw, "normalized_turns_dropped": dropped[i], "patch_matched": p is not None,
                     "eligible": c["repo"] not in excl and p is not None})
    elig = [r for r in rows if r["eligible"]]
    target = round(a.keep * len(elig))
    kept = set(select([(r["idx"], r["instance_id"], r["repo"], r["score"]) for r in elig], target,
                      a.max_per_repo, a.repo_penalty, a.instance_penalty))
    out = Path(a.out)
    stem = out.with_suffix("")
    with open(out, "w", encoding="utf-8") as fo:
        for r, c in zip(rows, convs):
            r["kept"] = r["idx"] in kept
            if r["kept"]:
                fo.write(json.dumps(c) + "\n")
    with open(str(stem) + ".scores.jsonl", "w", encoding="utf-8") as fs:
        for r in rows:
            fs.write(json.dumps(r) + "\n")
    K = [r for r in rows if r["kept"]]
    R = [r for r in elig if not r["kept"]]
    repos_all = collections.Counter(r["repo"] for r in elig)
    repos_kept = collections.Counter(r["repo"] for r in K)
    inst_kept = collections.Counter(r["instance_id"] for r in K)
    summary = {
        "params": {"keep": a.keep, "max_per_repo": a.max_per_repo, "repo_penalty": a.repo_penalty,
                   "instance_penalty": a.instance_penalty, "exclude_repos": sorted(excl), "weights": WEIGHTS,
                   "normalize": not a.no_normalize, "long_thought_chars": LONG_THOUGHT_CHARS},
        "counts": {"input": len(rows), "excluded_repo": sum(r["repo"] in excl for r in rows),
                   "patch_unmatched": sum(not r["patch_matched"] for r in rows), "eligible": len(elig),
                   "target": target, "kept": len(K), "kept_instances": len(inst_kept),
                   "kept_instances_with_2": sum(1 for v in inst_kept.values() if v == 2),
                   "repos_eligible": len(repos_all), "repos_kept": len(repos_kept),
                   "repos_at_cap": sum(1 for v in repos_kept.values() if v >= a.max_per_repo),
                   "normalized_trajectories": sum(1 for d in dropped if d), "normalized_turns_dropped": sum(dropped)},
        "score": {"eligible": quantiles([r["score"] for r in elig]), "kept": quantiles([r["score"] for r in K]),
                  "rejected": quantiles([r["score"] for r in R]),
                  "kept_min": min((r["score"] for r in K), default=None)},
        "feature_means": {k: {"kept": _mean(K, lambda r: r["features"][k]),
                              "rejected": _mean(R, lambda r: r["features"][k])} for k in WEIGHTS},
        "raw_means": {k: {"kept": _mean(K, lambda r: float(r["raw"][k])),
                          "rejected": _mean(R, lambda r: float(r["raw"][k]))}
                      for k in ("turns", "calls", "redundant", "tool_errors", "forbidden", "failed_retries",
                                "long_thoughts", "verify_after_edit", "run_before_edit", "read_before_edit")},
        "patch": {"kept_src_changed_lines": quantiles([r["raw"]["patch"]["src_changed"] for r in K]),
                  "eligible_src_changed_lines": quantiles([r["raw"]["patch"]["src_changed"] for r in elig]),
                  "kept_all_changed_lines": quantiles([r["raw"]["patch"]["changed"] for r in K]),
                  "kept_leaves_root_or_build_files": sum(r["raw"]["patch"]["adds_root_file"] for r in K),
                  "eligible_leaves_root_or_build_files": sum(r["raw"]["patch"]["adds_root_file"] for r in elig),
                  "kept_edits_existing_test": sum(r["raw"]["patch"]["edits_existing_test"] for r in K),
                  "eligible_edits_existing_test": sum(r["raw"]["patch"]["edits_existing_test"] for r in elig),
                  "kept_with_forbidden_action": sum(1 for r in K if r["raw"]["forbidden"]),
                  "eligible_with_forbidden_action": sum(1 for r in elig if r["raw"]["forbidden"])},
        "repos_kept_top": repos_kept.most_common(15),
        "repos_kept_histogram": dict(sorted(collections.Counter(repos_kept.values()).items())),
    }
    with open(str(stem) + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    c = summary["counts"]
    print(f"kept {c['kept']} of {c['eligible']} eligible ({c['input']} input, {c['excluded_repo']} excluded repo, "
          f"{c['patch_unmatched']} unmatched patch); {c['kept_instances']} instances ({c['kept_instances_with_2']} with 2), "
          f"{c['repos_kept']} of {c['repos_eligible']} repos; score kept min {summary['score']['kept_min']} "
          f"median {summary['score']['kept'].get('p50')} vs rejected median {summary['score']['rejected'].get('p50')}; "
          f"normalized {c['normalized_trajectories']} trajectories ({c['normalized_turns_dropped']} turns dropped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
