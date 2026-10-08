"""g4kit-render-distill: render converted trajectories (g4kit-convert-openhands) into LoRA training windows that
look exactly like the scorer's requests: Gemma 4 chat template with vLLM's preprocessing, the agent's system
prompt with the issue injected, the harness's first user message, the harness's tool declarations and the
scorer's tool-result encoding, with loss on every assistant turn.

Each output line is ``{"segments": [[text, train], ...], "instance_id", "turns"}``; ``train=1`` marks the
model's own output (thought, tool call and the ``<|tool_response>`` stop token, or the final text and
``<turn|>``). Trajectories longer than ``--max-len`` tokens are split into windows: system + user message +
consecutive turns, where a window after the first starts two turns before its first trained turn (context
only), roughly the raw tail that ADK compaction keeps.

  --mode think     teacher reasoning (text and think-tool thoughts) in the thought channel (thinking on)
  --mode nothink   tool calls only; the first model turn follows the empty thought block of the thinking-off prompt
  --encoding single  one JSON level, ensure_ascii False (the scorer since 2026-09-30)
  --encoding double  a JSON string wrapped as {"result": ...} (the harness before 2026-09-30)
  --mask-error-turns off   every assistant turn is trained (the default; output unchanged)
                     tool  no loss on an assistant turn whose tool call returned a tool error as g4kit-curate defines
                           it (file-tool errors, shell invocation errors, pytest usage errors); failing tests and
                           reproduction runs stay trained, as in SWE-Lego's error masking
                     all   no loss on any turn whose tool result has status error
                     A masked turn stays in the context; a window left without a trained turn is not written.

Inputs you provide (none ship with the kit): the system prompt of your agent (``{problem_description}`` is
replaced by the issue, as the harness's instruction templating does), the harness's first-user-message template
(``g4kit-harness user-template`` writes it from the installed harness), the tool declarations (a JSON list, or
taken from a logged request with ``--tools-from-log``), the chat template and a tokenizer for window lengths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable, Collection

from .. import assets
from ..chat import ChatRenderer, to_conversation
from ..toolcalls import EMPTY_THOUGHT
from .curate import tool_error

MODEL_HEAD = "<|turn>model\n"


def load_user_template(path: str) -> dict:
    tpl = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("user_message", "tree_section"):
        if key not in tpl:
            raise ValueError(f"user template {path} lacks {key!r}")
    tpl.setdefault("placeholders", {"repo": "<<REPO>>", "problem_statement": "<<PROBLEM_STATEMENT>>",
                                    "workspace_tree": "<<WORKSPACE_TREE>>"})
    return tpl


def tools_from_log(log_path: str, names: list[str]) -> list[dict]:
    """Declarations of ``names`` (in that order) from the first logged request (g4kit-proxy *_full.jsonl) that
    declares all of them. A subset is allowed: a single-agent design can reuse a coder request's declarations."""
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            tools = (json.loads(line).get("body") or {}).get("tools") or []
            by_name = {t.get("function", {}).get("name"): t for t in tools}
            if all(n in by_name for n in names):
                return [by_name[n] for n in names]
    raise ValueError(f"no request in {log_path} declares all of {names}")


def tree_from(msgs: list[dict]) -> str:
    """Workspace layout block, rebuilt from the first directory listing (the harness shows find -maxdepth 3)."""
    for m in msgs:
        if m["role"] == "tool" and m.get("name") == "run_command":
            out = json.loads(m["content"]).get("stdout", "")
            if out.startswith("/workspace/"):
                rows = sorted({"./" + ln[len("/workspace/"):].rstrip("/") for ln in out.split("\n")
                               if ln.startswith("/workspace/") and ln != "/workspace/" and "/." not in ln})
                return "\n".join(["."] + rows[:149])
            break
    return ""


def user_message(conv: dict, tpl: dict) -> str:
    ph = tpl["placeholders"]
    tree = tree_from(conv["messages"])
    text = tpl["user_message"] + (tpl["tree_section"].replace(ph["workspace_tree"], tree) if tree else "")
    return text.replace(ph["repo"], conv["repo"]).replace(ph["problem_statement"], conv["issue"])


def build_messages(conv: dict, system: str, tpl: dict, mode: str, encoding: str) -> list[dict]:
    msgs = [{"role": "system", "content": system.replace("{problem_description}", conv["issue"])},
            {"role": "user", "content": user_message(conv, tpl)}]
    for m in conv["messages"]:
        m = dict(m)
        if m["role"] == "tool":
            if encoding == "double":
                m["content"] = json.dumps({"result": m["content"]}, ensure_ascii=False)
            else:
                m["content"] = json.dumps(json.loads(m["content"]), ensure_ascii=False)
            m.pop("name", None)
        else:
            reasoning = m.pop("reasoning", None)
            if mode == "think" and reasoning:
                m["reasoning"] = reasoning
        msgs.append(m)
    return msgs


def error_turns(conv: dict, how: str) -> set[int]:
    """Indices into conv["messages"] of assistant turns to leave untrained (--mask-error-turns)."""
    if how == "off":
        return set()
    msgs, out = conv["messages"], set()
    for k, m in enumerate(msgs):
        if m["role"] != "assistant" or not m.get("tool_calls"):
            continue
        calls = {tc["id"]: tc["function"] for tc in m["tool_calls"]}
        for r in msgs[k + 1:]:
            if r["role"] != "tool":
                break
            f = calls.get(r["tool_call_id"])
            res = json.loads(r["content"])
            if f is None or res.get("status") != "error":
                continue
            if how == "all" or tool_error(f["name"], f["arguments"], res):
                out.add(k)
    return out


def spans(renderer: ChatRenderer, msgs: list[dict], tools: list[dict], mode: str,
          untrained: Collection[int] = frozenset()) -> list[tuple[str, int]]:
    """[(text, train)] covering the fully rendered conversation; assistant messages whose index is in
    ``untrained`` get train=0."""
    def render(ms: list[dict]) -> str:
        return renderer.render_conversation(to_conversation(ms, renderer.content_format), tools,
                                            add_generation_prompt=False, enable_thinking=(mode == "think"))

    full = render(msgs)
    out: list[tuple[str, int]] = []
    pos, first = 0, True
    for k, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        a, b = len(render(msgs[:k])), len(render(msgs[:k + 1]))
        if not full.startswith(render(msgs[:k + 1])):
            raise ValueError("chat template output is not prefix-stable for this conversation")
        seg = full[a:b]
        head = ""
        if seg.startswith(MODEL_HEAD):
            head, seg = MODEL_HEAD, seg[len(MODEL_HEAD):]
            if first and mode == "nothink":
                head += EMPTY_THOUGHT
        first = False
        out.append((full[pos:a] + head, 0))
        if seg.endswith("<turn|>\n"):           # generation stops at <turn|>; the newline is template glue
            seg, b = seg[:-1], b - 1
        out.append((seg, 0 if k in untrained else 1))
        pos = b
    if pos < len(full):
        out.append((full[pos:], 0))
    return out


def windows(segs: list[tuple[str, int]], encode: Callable[[str], list[int]], max_len: int,
            ctx_turns: int = 2) -> list[tuple[list[list], int]]:
    """Split [(text, train)] into windows of at most max_len tokens.

    segs alternates context / target; segs[0] is the system + user header (+ first model-turn head).
    Returns [(segments, trained turn count)]; a window whose targets are all untrained is dropped.
    """
    toks = [encode(t) for t, _ in segs]
    header, hdr_len = segs[0], len(toks[0])
    units = [(i, i + 1 if i + 1 < len(segs) else None) for i in range(1, len(segs), 2)]
    ulen = [len(toks[t]) + (len(toks[c]) if c is not None else 0) for t, c in units]

    def fill(start: int) -> int:
        n, e = hdr_len, start
        while e < len(units) and n + ulen[e] <= max_len:
            n += ulen[e]
            e += 1
        return e

    res: list[tuple[list[list], int]] = []
    s = 0
    while s < len(units):
        start = max(0, s - ctx_turns) if res else 0
        e = fill(start)
        if e <= s and start < s:
            start = s - 1
            e = fill(start)
        if e <= s:            # the next trained unit does not fit even with minimal context
            break
        w: list[list] = [[header[0], 0]]
        trained = 0
        for u in range(start, e):
            t, c = units[u]
            on = 1 if u >= s and segs[t][1] else 0
            trained += on
            w.append([segs[t][0], on])
            if c is not None and u < e - 1:
                w.append([segs[c][0], 0])
        if trained:           # with --mask-error-turns every target can be masked: nothing to learn from the window
            res.append((w, trained))
        s = e
    return res


def is_valid(instance_id: str, frac: float) -> bool:
    return int(hashlib.md5(instance_id.encode()).hexdigest(), 16) % 1000 < frac * 1000


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="g4kit-render-distill", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("conv", help="JSONL from g4kit-convert-openhands")
    ap.add_argument("out", help="output directory (train.jsonl, valid.jsonl)")
    ap.add_argument("--system-prompt", required=True, help="your agent's instruction text")
    ap.add_argument("--user-template", required=True, help="JSON from 'g4kit-harness user-template'")
    ap.add_argument("--tools", help="JSON list of tool declarations as the harness sends them")
    ap.add_argument("--tools-from-log", help="g4kit-proxy *_full.jsonl to take the tool declarations from")
    ap.add_argument("--tool-names", default="run_command,read_file,edit_file,write_file,get_status,submit_patch",
                    help="with --tools-from-log: the tool set (and order) of the agent being trained")
    ap.add_argument("--template", help="chat_template.jinja (default: $G4KIT_CHAT_TEMPLATE or the asset dir)")
    ap.add_argument("--tokenizer", required=True, help="Hugging Face tokenizer dir, tokenizer.json or repo id")
    ap.add_argument("--mode", choices=["think", "nothink"], default="nothink")
    ap.add_argument("--encoding", choices=["single", "double"], default="single")
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--valid-frac", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exclude-repos", default="", help="comma list, e.g. repositories of your local eval tasks")
    ap.add_argument("--mask-error-turns", choices=["off", "tool", "all"], default="off",
                    help="no loss on assistant turns whose tool call returned an error: tool = errors caused by the "
                         "call (g4kit-curate's rule), all = any error result")
    a = ap.parse_args(argv)
    if bool(a.tools) == bool(a.tools_from_log):
        raise SystemExit("give exactly one of --tools and --tools-from-log")
    tools = (json.loads(Path(a.tools).read_text(encoding="utf-8")) if a.tools
             else tools_from_log(a.tools_from_log, a.tool_names.split(",")))
    encode = _encoder(a.tokenizer)
    renderer = ChatRenderer.from_file(str(assets.resolve_template(a.template)))
    system = Path(a.system_prompt).read_text(encoding="utf-8")
    tpl = load_user_template(a.user_template)
    excl = set(filter(None, a.exclude_repos.split(",")))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n_traj = n_win = n_turn = n_masked = 0
    with open(out / "train.jsonl", "w", encoding="utf-8") as ft, open(out / "valid.jsonl", "w", encoding="utf-8") as fv, \
            open(a.conv, encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            if a.limit and i >= a.limit:
                break
            conv = json.loads(line)
            if conv["repo"] in excl:
                continue
            untrained = {k + 2 for k in error_turns(conv, a.mask_error_turns)}   # build_messages adds system + user
            n_masked += len(untrained)
            segs = spans(renderer, build_messages(conv, system, tpl, a.mode, a.encoding), tools, a.mode, untrained)
            dest = fv if is_valid(conv["instance_id"], a.valid_frac) else ft
            for w, nt in windows(segs, encode, a.max_len):
                dest.write(json.dumps({"segments": w, "instance_id": conv["instance_id"], "turns": nt},
                                      ensure_ascii=False) + "\n")
                n_win += 1
                n_turn += nt
            n_traj += 1
    print(f"{n_traj} trajectories -> {n_win} windows, {n_turn} trained turns ({a.mode}, {a.encoding}, max_len {a.max_len})"
          + (f"; {n_masked} error turns untrained ({a.mask_error_turns})" if a.mask_error_turns != "off" else ""))
    return 0


def _encoder(spec: str) -> Callable[[str], list[int]]:
    """Token ids without added special tokens (segments are concatenated by the trainer)."""
    try:
        from transformers import AutoTokenizer  # type: ignore

        path = Path(spec).expanduser()
        tok = AutoTokenizer.from_pretrained(str(path.parent if path.is_file() else path) if path.exists() else spec)
        return lambda text: tok.encode(text, add_special_tokens=False)
    except ImportError:
        from tokenizers import Tokenizer  # type: ignore

        path = Path(spec).expanduser()
        if path.is_dir():
            path = path / "tokenizer.json"
        tk = Tokenizer.from_file(str(path)) if path.exists() else Tokenizer.from_pretrained(spec)
        return lambda text: tk.encode(text, add_special_tokens=False).ids


if __name__ == "__main__":
    sys.exit(main())
