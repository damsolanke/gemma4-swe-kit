import json

from gemma4_swe_kit import logstats
from gemma4_swe_kit.toolcalls import STRING_DELIM as D

TOOLS = [{"type": "function", "function": {"name": n}} for n in ("run_command", "read_file", "edit_file")]


def turn(name, args, i, result):
    return [{"role": "assistant", "content": None,
             "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": json.dumps(result)}]


def synthetic_log():
    coder_hist, editor_hist = [], []
    for i in range(12):   # the coder re-runs one probe 12 times
        coder_hist += turn("run_command", {"command": "python -c 'probe()'"}, i, {"status": "ok", "stdout": "None"})
    editor_hist += turn("edit_file", {"filepath": "a.py", "old_string": "x", "new_string": "y"}, 0,
                        {"status": "error", "error_type": "FileEditError", "error_message": "old_string not found"})
    editor_hist += turn("edit_file", {"filepath": "a.py", "old_string": "x = 1", "new_string": "y"}, 1, {"status": "ok"})
    base = [{"role": "user", "content": "Issue 7"}]
    entries = []
    for k in range(1, 13):
        entries.append({"status": 200, "raw": "", "body": {"tools": TOOLS, "messages":
                        [{"role": "system", "content": "You are the coder."}] + base + coder_hist[: 2 * k]}})
    entries.append({"status": 200, "raw": "<|tool_call>call:grep -rn x .<tool_call|>", "body": {"tools": TOOLS, "messages":
                    [{"role": "system", "content": "You are the editor."}] + base + editor_hist}})
    entries.append({"status": 400, "error": "context_overflow", "raw": "", "body": {"tools": TOOLS, "messages":
                    [{"role": "system", "content": "You are the editor."}] + base + editor_hist}})
    entries.append({"status": 200, "raw": f"<|tool_call>call:read_file{{filepath:{D}a.py{D}}}<tool_call|>",
                    "body": {"tools": TOOLS, "messages": [{"role": "system", "content": "You are the editor."}]
                             + [{"role": "user", "content": "Issue 8"}]}})
    return entries


def test_sessions_loops_edits_and_errors():
    role = logstats.make_role_fn(["coder=coder", "editor=editor"])
    sessions = logstats.build_sessions(synthetic_log(), role)
    coder = sessions[("coder", "Issue 7")]
    editor = sessions[("editor", "Issue 7")]
    assert coder.requests == 12 and coder.longest_run() == 12 and coder.most_repeated() == 12
    assert editor.edit_outcomes() == {"not_found": 1, "ok": 1}
    assert (editor.malformed, editor.unparsed, editor.overflow) == (1, 1, 1)
    assert len(sessions) == 3
    text = logstats.report(sessions, loop_threshold=10)
    assert "coder" in text and "most repetitive sessions" in text


def test_cli(tmp_path, capsys):
    log = tmp_path / "proxy_full.jsonl"
    log.write_text("".join(json.dumps(e) + "\n" for e in synthetic_log()))
    assert logstats.main([str(log), "--role", "coder=coder"]) == 0
    assert "You are the editor." in capsys.readouterr().out
