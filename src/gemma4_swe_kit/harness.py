"""g4kit-harness: run the official competition harness locally the way the scorer runs it.

Must run inside the harness environment (Python 3.12 with the competition wheelhouse: swegemma,
adk_submission, adk_eval_core, google-adk); this module imports them lazily.

``run`` differs from ``swegemma eval`` in what the scorer does and the official CLI does not:

* the submission's ``eval_config.yaml`` budgets are applied (``evaluation:`` key or top level);
* ADK events compaction is on (``compaction_interval`` 5, ``token_threshold`` 14,336, ``overlap_size`` 2,
  ``event_retention_size`` 5, as the harness README states; the hosts' notebook uses interval 15) together with
  the context cache config (2,048 tokens, 1,800 s, 10 intervals); ``swegemma eval`` leaves both off;
* models are registered with ``setup_gemma_model_registry`` exactly as the scorer does, so the model string is
  ``openai/gemma-4-31b-it-qat-w4a16-ct`` (a name containing "gemma4" would switch tool results to the
  ``tool_responses`` role) and LoRA adapters route to ``openai/<adapter name>``;
* each task's start and end times are recorded for ``g4kit-scorer-time``.

``user-template`` writes the harness's first user message as a template for ``g4kit-render-distill``, generated
from the installed harness so it always matches its version.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

SCORER_MODEL = "gemma-4-31b-it-qat-w4a16-ct"
REPO, PROBLEM, TREE = "<<REPO>>", "<<PROBLEM_STATEMENT>>", "<<WORKSPACE_TREE>>"


def load_eval_config(sub_dir: Path) -> dict:
    path = sub_dir / "eval_config.yaml"
    if not path.exists():
        return {}
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return raw.get("evaluation", raw) or {}


def use_task_env(task_env: str) -> None:
    """Subprocess sandboxes see the host's packages through a .pth file. Locally the host is the harness venv,
    so point sandbox venvs at a separate environment that holds the task repositories' dependencies."""
    from swegemma.sandbox.subprocess import SubprocessManager

    site = sorted(Path(os.path.expanduser(task_env)).glob("lib/python*/site-packages"))
    if not site:
        raise SystemExit(f"--task-env {task_env}: no lib/python*/site-packages found")
    orig = SubprocessManager.start

    def start(self, *args, **kwargs):
        sid = orig(self, *args, **kwargs)
        for sp in Path(self._sandboxes[sid]["venv"]).glob("lib/python*/site-packages"):
            (sp / "_host_env.pth").write_text(str(site[0]) + "\n", encoding="utf-8")
        return sid

    SubprocessManager.start = start


def compaction_config(interval: int, threshold: int):
    try:
        from google.adk.apps._configs import EventsCompactionConfig
    except ImportError:
        from google.adk.apps.app import EventsCompactionConfig
    return EventsCompactionConfig(compaction_interval=interval, overlap_size=2, token_threshold=threshold,
                                  event_retention_size=5)


async def run_arm(name: str, sub_dir: Path, tasks: list, a: argparse.Namespace, out_dir: Path) -> None:
    from adk_submission import discover_adapters
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits
    from swegemma.evaluate import Evaluator
    from swegemma.models import setup_gemma_model_registry

    ev = load_eval_config(sub_dir)
    manifest = discover_adapters(str(sub_dir), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    models = setup_gemma_model_registry(api_base=a.api_base, api_key=a.api_key, adapter_manifest=manifest)
    limits, gen = build_submission_limits()
    minutes = ev.get("max_time_minutes")
    cfg = EvalConfig(
        tasks_path=Path(a.tasks), snapshots_dir=Path(a.snapshots), results_dir=out_dir / f"results_{name}",
        submission_dir=sub_dir, models=models, sandbox=a.sandbox, image=a.image,
        timeout_seconds=ev.get("timeout_seconds"),
        max_time_minutes=float(minutes) * a.time_scale if minutes is not None else None,
        max_tool_calls=ev.get("max_tool_calls"), max_turns=ev.get("max_turns"),
        limits=limits, generation_constraints=gen, adapter_manifest=manifest,
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
        events_compaction_config=compaction_config(a.compaction_interval, a.compaction_threshold),
        graph_dir=a.graph_dir or "data/graphs", embeddings_dir=a.embeddings_dir or "data/embeddings",
        wheels_dir=Path(a.wheels_dir) if a.wheels_dir else None, verbose=False, display_mode="quiet",
    )
    evaluator = Evaluator(cfg)
    out = out_dir / f"results_{name}.jsonl"
    done = {json.loads(line)["id"] for line in open(out, encoding="utf-8")} if out.exists() else set()
    for i, task in enumerate(tasks, 1):
        if task.instance_id in done:
            continue
        t0 = time.time()
        try:
            r = await evaluator.evaluate_task(task=task, task_index=i, total_tasks=len(tasks))
            rec = {"arm": name, "id": task.instance_id, "repo": task.repo, "resolved": bool(r.resolved),
                   "started_at": round(t0, 3), "ended_at": round(time.time(), 3), "wall_s": round(time.time() - t0, 1),
                   "tool_calls": r.tool_calls, "status": r.status, "error": (r.error_message or "")[:800],
                   "test_exit_code": r.test_exit_code, "patch_chars": len(r.agent_patch or ""),
                   "patch": (r.agent_patch or "")[:20000], "test_output_tail": (r.test_output or "")[-3000:]}
        except Exception as exc:
            rec = {"arm": name, "id": task.instance_id, "repo": task.repo, "resolved": False,
                   "started_at": round(t0, 3), "ended_at": round(time.time(), 3), "wall_s": round(time.time() - t0, 1),
                   "error": f"EXC {type(exc).__name__}: {exc}"[:800]}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{time.strftime('%H:%M:%S')} {name} [{i}/{len(tasks)}] {task.instance_id} resolved={rec['resolved']} "
              f"wall={rec['wall_s']}s calls={rec.get('tool_calls')} patch={rec.get('patch_chars')} "
              f"{(rec.get('error') or '')[:150]}", flush=True)


def cmd_run(a: argparse.Namespace) -> int:
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    import litellm
    from swegemma.models import load_tasks

    litellm.drop_params = True
    if a.task_env:
        use_task_env(a.task_env)
    snapshots = Path(a.snapshots)
    by_id = {t.instance_id: t for t in load_tasks(Path(a.tasks))}
    ids = a.ids or list(by_id)[: a.n]
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"unknown task ids: {missing}")
    tasks = [by_id[i] for i in ids if (snapshots / f"{i}.tgz").exists() or not a.require_snapshot]
    out_dir = Path(a.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"g4kit-harness -> {out_dir} | {len(tasks)} tasks | model endpoint {a.api_base} | compaction interval "
          f"{a.compaction_interval}, threshold {a.compaction_threshold}", flush=True)
    for spec in a.arm:
        name, sep, path = spec.partition("=")
        sub_dir = Path(path if sep else name).expanduser().resolve()
        name = name if sep else sub_dir.name
        asyncio.run(run_arm(name, sub_dir, tasks, a, out_dir))
    return 0


def user_template(time_minutes: float | None, tool_calls: int | None, turns: int | None,
                  code_intel: bool = False) -> dict:
    """The harness's first user message with placeholders, built by the installed swegemma."""
    import tempfile

    from swegemma.budget import EvaluationBudget, HarnessLimits
    from swegemma.harness.agent_runner import build_agent_prompt

    budget = EvaluationBudget(time_minutes=time_minutes, tool_calls=tool_calls, turns=turns)
    with tempfile.TemporaryDirectory() as tmp:
        graph_dir = emb_dir = None
        if code_intel:   # the code-intelligence section appears when graph data exists for the repo
            graph_dir, emb_dir = Path(tmp) / "g", Path(tmp) / "e"
            graph_dir.mkdir()
            emb_dir.mkdir()
            (graph_dir / "repo.json").write_text("x" * 200)
            (emb_dir / "repo.npz").write_bytes(b"x" * 200)
        config = SimpleNamespace(budget=budget, harness=HarnessLimits(), enable_sandbox_testing=True,
                                 graph_dir=str(graph_dir) if graph_dir else None,
                                 embeddings_dir=str(emb_dir) if emb_dir else None)
        task = SimpleNamespace(repo=REPO if not code_intel else "org/repo", problem_statement=PROBLEM, hints_text="",
                               base_commit=None)
        head = build_agent_prompt(task, config, "")
        with_tree = build_agent_prompt(task, config, TREE)
    if code_intel:
        head = head.replace("org/repo", REPO)
        with_tree = with_tree.replace("org/repo", REPO)
    if not with_tree.startswith(head):
        raise RuntimeError("unexpected prompt layout: the tree section is not a suffix")
    return {"user_message": head, "tree_section": with_tree[len(head):],
            "placeholders": {"repo": REPO, "problem_statement": PROBLEM, "workspace_tree": TREE}}


def cmd_user_template(a: argparse.Namespace) -> int:
    tpl = user_template(a.time_minutes, a.tool_calls, a.turns, a.code_intel)
    Path(a.out).write_text(json.dumps(tpl, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="g4kit-harness", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="evaluate submission(s) on tasks with the scorer's settings",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    r.add_argument("--arm", action="append", required=True, metavar="[NAME=]SUBMISSION_DIR",
                   help="submission directory (repeatable; arms run one after another on the same tasks)")
    r.add_argument("--tasks", required=True, help="tasks.jsonl")
    r.add_argument("--snapshots", required=True, help="directory of <instance_id>.tgz repository snapshots")
    r.add_argument("--ids", nargs="*", help="task ids (default: the first --n tasks of the file)")
    r.add_argument("--n", type=int, default=12)
    r.add_argument("--out", default=f"runs/{time.strftime('%Y%m%d_%H%M')}")
    r.add_argument("--api-base", default="http://127.0.0.1:11436/v1", help="g4kit-proxy, g4kit-fake-llm or a vLLM server")
    r.add_argument("--api-key", default="local")
    r.add_argument("--sandbox", choices=["subprocess", "docker"], default="subprocess")
    r.add_argument("--image", default="swebench-sandbox:latest", help="docker sandbox image")
    r.add_argument("--time-scale", type=float, default=1.0,
                   help="multiply max_time_minutes, to offset a slower local model (e.g. 3 on a laptop)")
    r.add_argument("--compaction-interval", type=int, default=5)
    r.add_argument("--compaction-threshold", type=int, default=14336)
    r.add_argument("--wheels-dir", help="wheels for subprocess sandboxes")
    r.add_argument("--graph-dir")
    r.add_argument("--embeddings-dir")
    r.add_argument("--task-env", help="venv holding task dependencies, exposed to subprocess sandboxes")
    r.add_argument("--require-snapshot", action=argparse.BooleanOptionalAction, default=True,
                   help="skip tasks without <id>.tgz in --snapshots")
    r.set_defaults(func=cmd_run)
    u = sub.add_parser("user-template", help="write the harness's first user message as a template (JSON)")
    u.add_argument("--out", default="user_template.json")
    u.add_argument("--time-minutes", type=float, default=8.0)
    u.add_argument("--tool-calls", type=int, default=100)
    u.add_argument("--turns", type=int, default=500)
    u.add_argument("--code-intel", action="store_true", help="include the code-intelligence tools section")
    u.set_defaults(func=cmd_user_template)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
