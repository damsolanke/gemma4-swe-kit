"""g4kit-fake-llm: a scripted OpenAI-compatible model for smoke-testing a submission through the real harness
in seconds, without a GPU.

Each agent session gets a fixed sequence of tool calls. In the default "auto" script, a session whose request
declares submit_patch is treated as the main agent: it reads a file, runs a command, calls every sub-agent tool
once (tools whose only parameter is ``request``, as ADK's AgentTool declares them), checks get_status, submits,
and ends with a text reply. Other sessions (sub-agents) read, run a command and answer "OK". Steps for tools
that a request does not declare are skipped. A JSON script can override this per role.

Every request is logged with the request's sampling settings and the last tool result it carries, and the
summary reports what a smoke test should confirm: which roles ran, their thinking/max_tokens/temperature
settings as they reached the server, the tool-result encoding (single-level JSON since the 2026-09-30 harness),
the role used for tool results ("tool"; "tool_responses" means the model name contains "gemma4"), and tool
errors.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HARNESS_TOOLS = {"run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch",
                 "search_similar_code", "get_code_neighbors", "get_code_subgraph"}


def declared_tools(body: dict) -> dict[str, dict]:
    return {t["function"]["name"]: t["function"] for t in body.get("tools") or [] if t.get("function", {}).get("name")}


def is_agent_tool(fn: dict) -> bool:
    props = ((fn.get("parameters") or {}).get("properties") or {})
    return set(props) == {"request"}


def system_text(body: dict) -> str:
    msgs = body.get("messages") or []
    if not msgs or msgs[0].get("role") not in ("system", "developer"):
        return ""
    c = msgs[0].get("content")
    return c if isinstance(c, str) else " ".join(p.get("text", "") for p in c or [] if isinstance(p, dict))


def tool_result_encoding(content: Any) -> str:
    """'single' (one JSON level), 'double' (a JSON string wrapped in {"result": ...}), 'raw' text, or 'parts'."""
    if isinstance(content, list):
        return "parts"
    if not isinstance(content, str):
        return "other"
    try:
        obj = json.loads(content)
    except ValueError:
        return "raw"
    if isinstance(obj, dict) and set(obj) == {"result"} and isinstance(obj["result"], str):
        try:
            inner = json.loads(obj["result"])
        except ValueError:
            return "single"
        return "double" if isinstance(inner, (dict, list)) else "single"
    return "single"


def auto_steps(tools: dict[str, dict], read_file: str) -> tuple[str, list[dict]]:
    if "submit_patch" in tools:
        steps = [{"tool": "read_file", "args": {"filepath": read_file, "start_line": 1, "end_line": 5}},
                 {"tool": "run_command", "args": {"command": "ls | head -5"}}]
        steps += [{"tool": name, "args": {"request": "Smoke test: reply OK in one line."}}
                  for name, fn in tools.items() if name not in HARNESS_TOOLS and is_agent_tool(fn)]
        steps += [{"tool": "get_status", "args": {}}, {"tool": "submit_patch", "args": {}},
                  {"text": "Done: patch submitted."}]
        return "main", steps
    return "sub", [{"tool": "read_file", "args": {"filepath": read_file, "start_line": 1, "end_line": 3}},
                   {"tool": "run_command", "args": {"command": "ls | head -3"}}, {"text": "OK"}]


class FakeLLM:
    def __init__(self, script: dict | None = None, read_file: str = "README.md", log_path: str | None = None):
        self.roles = [(r.get("name", f"role{i}"), re.compile(r["match"]), r["steps"])
                      for i, r in enumerate((script or {}).get("roles", []))]
        self.read_file = read_file
        self.log_path = Path(log_path) if log_path else None
        self.records: list[dict] = []
        self._lock = threading.Lock()

    def plan(self, body: dict) -> tuple[str, list[dict]]:
        sys_text = system_text(body)
        for name, pattern, steps in self.roles:
            if pattern.search(sys_text):
                return name, steps
        return auto_steps(declared_tools(body), self.read_file)

    def respond(self, body: dict) -> dict:
        msgs = body.get("messages") or []
        tools = declared_tools(body)
        role, steps = self.plan(body)
        steps = [s for s in steps if "text" in s or s["tool"] in tools]
        step = sum(1 for m in msgs if m.get("role") == "assistant" and m.get("tool_calls"))
        nxt = steps[step] if step < len(steps) else {"text": "OK"}
        tool_msgs = [m for m in msgs if m.get("role") not in ("system", "developer", "user", "assistant")]
        last_tool = tool_msgs[-1] if tool_msgs else None
        kw = body.get("chat_template_kwargs") or {}
        rec = {"t": round(time.time(), 3), "role": role, "step": step, "n_messages": len(msgs),
               "system_head": system_text(body)[:60], "tools": list(tools),
               "enable_thinking": kw.get("enable_thinking"), "thinking_token_budget": body.get("thinking_token_budget"),
               "max_tokens": body.get("max_completion_tokens") or body.get("max_tokens"),
               "temperature": body.get("temperature"), "top_k": body.get("top_k"), "model": body.get("model"),
               "tool_result_roles": sorted({m.get("role") for m in tool_msgs}),
               "last_tool_result": None if last_tool is None else str(last_tool.get("content"))[:600],
               "last_tool_encoding": None if last_tool is None else tool_result_encoding(last_tool.get("content"))}
        with self._lock:
            self.records.append(rec)
            if self.log_path:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if "text" in nxt:
            message: dict[str, Any] = {"role": "assistant", "content": nxt["text"], "reasoning": None, "tool_calls": []}
            finish = "stop"
        else:
            message = {"role": "assistant", "content": None, "reasoning": None,
                       "tool_calls": [{"id": "chatcmpl-tool-" + uuid.uuid4().hex, "type": "function",
                                       "function": {"name": nxt["tool"],
                                                    "arguments": json.dumps(nxt.get("args") or {})}}]}
            finish = "tool_calls"
        return {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
                "model": body.get("model", "fake"),
                "choices": [{"index": 0, "message": message, "logprobs": None, "finish_reason": finish}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}}


def summarize(records: list[dict]) -> str:
    if not records:
        return "no requests received"
    by_role: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        by_role[(r["role"], r.get("system_head", "")[:40])].append(r)
    lines = [f"{len(records)} requests, {len(by_role)} agents"]
    for (role, head), rs in by_role.items():
        settings = Counter((r.get("enable_thinking"), r.get("thinking_token_budget"), r.get("max_tokens"),
                            r.get("temperature"), r.get("top_k")) for r in rs)
        lines.append(f"  {role} {head!r}: {len(rs)} requests")
        lines.append(f"    tools: {', '.join(rs[0]['tools'])}")
        for (think, budget, max_tok, temp, top_k), n in settings.most_common():
            lines.append(f"    enable_thinking={think} thinking_token_budget={budget} max_tokens={max_tok} "
                         f"temperature={temp} top_k={top_k}  ({n} requests)")
    enc = Counter(r["last_tool_encoding"] for r in records if r.get("last_tool_encoding"))
    roles = Counter(role for r in records for role in r.get("tool_result_roles") or [])
    errors = sum(1 for r in records if r.get("last_tool_result") and '"status": "error"' in r["last_tool_result"])
    lines.append(f"  tool-result encoding: {dict(enc)} (expected: single)")
    lines.append(f"  tool-result message roles: {dict(roles)} (expected: tool)")
    lines.append(f"  tool results with status error: {errors}")
    if roles.get("tool_responses"):
        lines.append("  WARNING: tool results use role tool_responses: the model name contains 'gemma4', unlike the scorer")
    return "\n".join(lines)


def make_handler(fake: FakeLLM):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def _send(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self._send(200, {"object": "list", "data": [{"id": "fake", "object": "model"}]})

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            self._send(200, fake.respond(body))

    return Handler


def serve(fake: FakeLLM, host: str = "127.0.0.1", port: int = 11450) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(fake))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="g4kit-fake-llm", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11450)
    ap.add_argument("--script", help='JSON: {"roles": [{"name", "match": regex on the system prompt, "steps": '
                                     '[{"tool", "args"} | {"text"}]}]}; unmatched sessions use the auto script')
    ap.add_argument("--read-file", default="README.md", help="file the auto script reads")
    ap.add_argument("--log", default="g4kit_fake_llm.jsonl")
    ap.add_argument("--report", metavar="LOG", help="print the summary of an existing log and exit")
    a = ap.parse_args(argv)
    if a.report:
        with open(a.report, encoding="utf-8") as f:
            print(summarize([json.loads(line) for line in f if line.strip()]))
        return 0
    script = json.loads(Path(a.script).read_text(encoding="utf-8")) if a.script else None
    fake = FakeLLM(script, a.read_file, a.log)
    server = serve(fake, a.host, a.port)
    print(f"g4kit-fake-llm on http://{a.host}:{server.server_address[1]}/v1 (log {a.log}); Ctrl-C prints the summary",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print(summarize(fake.records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
