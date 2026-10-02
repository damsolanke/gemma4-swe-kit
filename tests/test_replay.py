"""Replay harness: context selection, request edits, metrics and paired summaries, against a fake server."""
import json
import random
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from conftest import ServerThread

from gemma4_swe_kit.replay import conditions, metrics, run
from gemma4_swe_kit.toolcalls import STRING_DELIM as D

TOOLS = [{"type": "function", "function": {"name": n, "parameters": {}}}
         for n in ("run_command", "read_file", "edit_file", "submit_patch")]
BACKTICK_PROMPT = "You fix bugs. Search with `grep -rn NAME . | head -40`."
GOOD = f"<|tool_call>call:run_command{{command:{D}grep -rn widen .{D}}}<tool_call|>"
BAD = "<|tool_call>call:grep -rn widen . | head -40<tool_call|>"


def entry(system, history, raw):
    return {"t": 0, "status": 200, "raw": raw, "body": {
        "model": "gemma-4-31b-it-qat-w4a16-ct", "tools": TOOLS, "temperature": 0.6,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": "Issue."}] + history}}


def call(name, args, i, result='{"status": "ok"}'):
    return [{"role": "assistant", "content": None, "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]},
        {"role": "tool", "tool_call_id": f"c{i}", "content": result}]


def sample(raw_message, raw=None, prev=None, body=None):
    return metrics.Sample(body=body or {"tools": TOOLS}, message=raw_message, finish="stop", raw=raw, prev_call=prev)


def test_metrics():
    bad = sample({"content": BAD, "tool_calls": []}, raw=BAD)
    assert metrics.malformed_name(bad) and metrics.unparsed_call(bad) and metrics.no_tool_call(bad)
    assert not metrics.text_only(bad)
    via_vllm = sample({"content": BAD, "tool_calls": []})          # a real vLLM server returns no raw text
    assert metrics.malformed_name(via_vllm)
    unknown = sample({"content": None, "tool_calls": [{"function": {"name": "grep", "arguments": "{}"}}]})
    assert metrics.unknown_tool(unknown) and metrics.malformed_name(unknown)
    rep = sample({"content": None, "tool_calls": [{"function": {"name": "read_file", "arguments": '{"filepath": "a.py"}'}}]},
                 prev=("read_file", json.dumps({"filepath": "a.py"}, sort_keys=True)))
    assert metrics.exact_repeat(rep) and not metrics.malformed_name(rep)
    assert metrics.text_only(sample({"content": "I am done.", "tool_calls": []}))


def test_conditions(tmp_path):
    subs = tmp_path / "subs.json"
    subs.write_text(json.dumps([[r"Search with `grep -rn NAME \. \| head -40`\.",
                                 "Search by calling run_command with a recursive grep piped to head -40."]]))
    c = conditions.parse(f"prose=subs:{subs},temp:0.2,think:on,budget:256,max_tokens:512")
    body, k = c.apply(entry(BACKTICK_PROMPT, [], None)["body"])
    assert k == 1 and "`" not in body["messages"][0]["content"]
    assert (body["temperature"], body["chat_template_kwargs"], body["thinking_token_budget"], body["max_tokens"]) == \
        (0.2, {"enable_thinking": True}, 256, 512)
    assert conditions.parse("orig=").apply({"messages": []}) == ({"messages": []}, 0)


def test_selectors():
    entries = [
        entry(BACKTICK_PROMPT, [], BAD),                                                     # malformed onset
        entry(BACKTICK_PROMPT, [{"role": "assistant", "content": BAD},
                                {"role": "user", "content": "Your previous response reached the token limit before "
                                                            "the tool call finished closing"}], BAD),  # not an onset
        entry(BACKTICK_PROMPT, call("read_file", {"filepath": "a.py"}, 1) + call("read_file", {"filepath": "a.py"}, 2), GOOD),
        entry(BACKTICK_PROMPT, call("edit_file", {"filepath": "a.py"}, 1, '{"status": "error"}'), GOOD),
        entry(BACKTICK_PROMPT, call("read_file", {"filepath": "b.py"}, 1),
              f"<|tool_call>call:read_file{{filepath:{D}b.py{D}}}<tool_call|>"),
    ]
    assert len(run.select(entries, "all")) == 5
    assert len(run.select(entries, "malformed")) == 1
    assert len(run.select(entries, "malformed", onset=False)) == 2
    assert len(run.select(entries, "repeat")) == 1
    assert len(run.select(entries, "after-error")) == 1
    assert len(run.select(entries, "repeat-onset")) == 1
    assert run.sign_test(10, 0) < 0.01 and run.sign_test(3, 3) == 1.0


class PromptSensitiveModel(BaseHTTPRequestHandler):
    """Writes the shell command as the tool name 3 times in 4 when the system prompt shows a backticked
    command, and well-formed calls otherwise."""

    rnd = random.Random(0)

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        system = body["messages"][0]["content"]
        raw = BAD if ("`grep" in system and self.rnd.random() < 0.75) else GOOD
        content = raw if raw == BAD else None
        calls = [] if raw == BAD else [{"id": "x", "type": "function",
                                        "function": {"name": "run_command", "arguments": '{"command": "grep -rn widen ."}'}}]
        data = json.dumps({"choices": [{"message": {"role": "assistant", "content": content, "tool_calls": calls},
                                        "finish_reason": "tool_calls" if calls else "stop"}],
                           "usage": {"completion_tokens": 12}, "g4kit": {"raw": raw}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_end_to_end_prose_rewrite(tmp_path):
    log = tmp_path / "proxy_full.jsonl"
    log.write_text("".join(json.dumps(entry(BACKTICK_PROMPT, call("read_file", {"filepath": f"f{i}.py"}, i), BAD)) + "\n"
                           for i in range(6)))
    subs = tmp_path / "subs.json"
    subs.write_text(json.dumps([["`grep -rn NAME \\. \\| head -40`", "a recursive grep through run_command"]]))
    server = ThreadingHTTPServer(("127.0.0.1", 0), PromptSensitiveModel)
    with ServerThread(server) as srv:
        out = tmp_path / "replay.jsonl"
        assert run.main([str(log), "--select", "malformed", "--cond", "orig=", "--cond", f"prose=subs:{subs}",
                         "--n", "4", "--api-base", srv.url + "/v1", "--out", str(out), "--require-subs"]) == 0
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(recs) == 6 * 2 * 4
    rate = {c: sum(r["metrics"]["malformed_name"] for r in recs if r["cond"] == c) for c in ("orig", "prose")}
    assert rate["prose"] == 0 and rate["orig"] > 10
    text = run.summarize(recs, list(metrics.DEFAULT), ["orig", "prose"])
    assert re.search(r"paired prose vs orig, malformed_name: lower in \d+ contexts", text)


def test_example_rewrites_apply():
    root = Path(__file__).parents[1] / "examples"
    cond = conditions.parse(f"prose=subs:{root / 'prose_rewrites.json'}")
    body = {"messages": [{"role": "system", "content": (root / "system_prompt_with_commands.md").read_text()}]}
    new, k = cond.apply(body)
    assert k == 3 and "`" not in new["messages"][0]["content"]
