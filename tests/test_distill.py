"""OpenHands conversion on a hand-written trajectory, and training-window rendering with a mock tokenizer."""
import json

import pytest
from conftest import FIXTURES

from gemma4_swe_kit.chat import to_conversation
from gemma4_swe_kit.distill.convert import Conv, Skip, convert_rows, iter_rows
from gemma4_swe_kit.distill.render import (build_messages, error_turns, load_user_template, main, spans,
                                          tools_from_log, windows)
from gemma4_swe_kit.toolcalls import EMPTY_THOUGHT


@pytest.fixture
def converted():
    kept, why = convert_rows(iter_rows(str(FIXTURES / "openhands_tiny.jsonl")))
    return kept, why


def test_conversion_keeps_resolved_and_reports_skips(converted):
    kept, why = converted
    assert [c["instance_id"] for c in kept] == ["acme__widgets-42"]
    assert why == {"undo_edit": 1, "unresolved": 1}


def test_tool_mapping(converted):
    conv = converted[0][0]
    assert conv["issue"].startswith("widen() grows the width") and conv["n_turns"] == 4
    assistant = [m for m in conv["messages"] if m["role"] == "assistant"]
    calls = [(m["tool_calls"][0]["function"]["name"], m["tool_calls"][0]["function"]["arguments"])
             for m in assistant if m.get("tool_calls")]
    assert calls == [
        ("read_file", {"filepath": "widgets/core.py"}),
        ("edit_file", {"filepath": "widgets/core.py", "old_string": "    return width + abs(pad)",
                       "new_string": "    return width + max(pad, 0)"}),
        ("run_command", {"command": "cd /workspace && python -m pytest -q tests/test_core.py"}),
        ("submit_patch", {}),
    ]
    # the think-tool thought and the turn's text are folded into the reasoning of the next call
    assert assistant[0]["reasoning"] == "The bug should be in widen() in widgets/core.py.\n\nLet me read the function."
    assert assistant[-1] == {"role": "assistant", "content": "Submitted the fix.", "reasoning": None}
    results = {m["name"]: json.loads(m["content"]) for m in conv["messages"] if m["role"] == "tool"}
    assert results["read_file"]["content"] == "def widen(width, pad):\n    return width + abs(pad)\n"
    assert results["read_file"]["start_line"] == 1 and results["read_file"]["is_truncated"] is False
    assert "-    return width + abs(pad)\n+    return width + max(pad, 0)" in results["edit_file"]["diff"]
    assert results["run_command"] == {"status": "ok", "stdout": "..\n2 passed in 0.01s", "stderr": "", "exit_code": 0}
    assert results["submit_patch"]["files_changed"] == 1


def test_long_view_becomes_range_reads():
    row = json.loads((FIXTURES / "openhands_tiny.jsonl").read_text().splitlines()[0])
    conv = Conv(row, max_lines=1)
    lines = "\n".join(f"{i:6d}\tline {i}" for i in (1, 2, 3))
    obs = f"Here's the result of running `cat -n` on {conv.root}/widgets/core.py:\n{lines}\n"
    out = conv.view({"command": "view"}, obs, f"{conv.root}/widgets/core.py")
    assert [args for _, args, _ in out] == [{"filepath": "widgets/core.py", "start_line": i, "end_line": i} for i in (1, 2, 3)]
    with pytest.raises(Skip):
        Conv(row, max_lines=1).view({"command": "view"}, obs.replace("     3\tline 3\n", "     3\tline 3\n     4\tx\n"),
                                    f"{conv.root}/widgets/core.py")


def mock_encode(text):
    return text.split()


@pytest.mark.parametrize("mode", ["nothink", "think"])
def test_render_windows(converted, mini_renderer, tools, mode):
    conv = converted[0][0]
    system = (FIXTURES / "system_prompt.md").read_text()
    tpl = load_user_template(str(FIXTURES / "user_template.json"))
    msgs = build_messages(conv, system, tpl, mode, "single")
    assert "widen() grows the width" in msgs[0]["content"] and "Repository acme/widgets." in msgs[1]["content"]
    segs = spans(mini_renderer, msgs, tools, mode)
    trained = [t for t, flag in segs if flag]
    assert len(trained) == 5                                   # 4 tool calls + the final text
    assert trained[0].startswith("<|channel>thought\n") == (mode == "think")
    assert all(t.endswith("<tool_call|><|tool_response>") for t in trained[:4]) and trained[-1].endswith("<turn|>")
    # the segments tile the rendered conversation; nothink adds the empty thought the generation prompt ends with
    full = mini_renderer.render_conversation(to_conversation(msgs, "openai"), tools, add_generation_prompt=False,
                                             enable_thinking=(mode == "think"))
    expected = full.replace("<|turn>model\n", "<|turn>model\n" + EMPTY_THOUGHT, 1) if mode == "nothink" else full
    assert "".join(t for t, _ in segs) == expected
    one = windows(segs, mock_encode, max_len=10_000)
    assert len(one) == 1 and one[0][1] == 5


def test_windows_split_with_context_turns():
    """header 10 tokens; units (target + following context) of 10, 10 and 5 tokens; max_len 30."""
    segs = [("h " * 10, 0), ("t1 " * 5, 1), ("c1 " * 5, 0), ("t2 " * 5, 1), ("c2 " * 5, 0), ("t3 " * 5, 1)]
    out = windows(segs, mock_encode, max_len=30)
    assert [[flag for _, flag in w] for w, _ in out] == [[0, 1, 0, 1], [0, 0, 0, 1]]
    assert [n for _, n in out] == [2, 1]
    assert all(sum(len(mock_encode(t)) for t, _ in w) <= 30 for w, _ in out)
    assert windows(segs, mock_encode, max_len=12) == []          # nothing fits next to the header


def test_tool_encodings(converted):
    conv = converted[0][0]
    tpl = load_user_template(str(FIXTURES / "user_template.json"))
    single = build_messages(conv, "S", tpl, "nothink", "single")
    double = build_messages(conv, "S", tpl, "nothink", "double")
    s = next(m["content"] for m in single if m["role"] == "tool")
    d = next(m["content"] for m in double if m["role"] == "tool")
    assert json.loads(s)["status"] == "ok"
    assert json.loads(json.loads(d)["result"])["status"] == "ok"


def test_tools_from_log_subset(tmp_path, tools):
    log = tmp_path / "proxy_full.jsonl"
    log.write_text(json.dumps({"body": {"tools": tools + [{"type": "function", "function": {"name": "helper"}}]}}) + "\n")
    picked = tools_from_log(str(log), ["read_file", "run_command"])
    assert [t["function"]["name"] for t in picked] == ["read_file", "run_command"]


def conv_with_errors():
    """Calls whose results are: a read_file error (tool error), a failing test run (expected), an empty grep (expected),
    a missing command (tool error), then an edit and submit_patch."""
    steps = [("read_file", {"filepath": "nope.py"}, {"status": "error", "error_type": "FileReadError",
                                                     "error_message": "File not found: nope.py"}),
             ("run_command", {"command": "python -m pytest -q tests/test_core.py"},
              {"status": "error", "error_type": "CommandError", "error_message": "1 failed",
               "details": {"stdout": "1 failed", "stderr": "", "exit_code": 1}}),
             ("run_command", {"command": "grep -rn widen docs"},
              {"status": "error", "error_type": "CommandError", "error_message": "",
               "details": {"stdout": "", "stderr": "", "exit_code": 1}}),
             ("run_command", {"command": "pytset -q"},
              {"status": "error", "error_type": "CommandError", "error_message": "pytset: command not found",
               "details": {"stdout": "", "stderr": "pytset: command not found", "exit_code": 127}}),
             ("edit_file", {"filepath": "widgets/core.py", "old_string": "abs(pad)", "new_string": "max(pad, 0)"},
              {"status": "ok"}),
             ("submit_patch", {}, {"status": "ok", "patch_size": 10, "files_changed": 1})]
    msgs = []
    for i, (name, args, result) in enumerate(steps):
        msgs.append({"role": "assistant", "content": None, "reasoning": None,
                     "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": args}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "name": name, "content": json.dumps(result)})
    msgs.append({"role": "assistant", "content": "Submitted the fix.", "reasoning": None})
    return {"instance_id": "acme__widgets-7", "repo": "acme/widgets", "issue": "widen() grows on negative pad",
            "n_turns": len(steps), "messages": msgs}


def test_error_turn_rules():
    conv = conv_with_errors()
    assert error_turns(conv, "off") == set()
    assert error_turns(conv, "tool") == {0, 6}                 # the file-tool error and the missing command
    assert error_turns(conv, "all") == {0, 2, 4, 6}            # every result with status error


@pytest.mark.parametrize("how,trained", [("off", 7), ("tool", 5), ("all", 3)])
def test_masked_turns_stay_in_context(mini_renderer, tools, how, trained):
    conv = conv_with_errors()
    tpl = load_user_template(str(FIXTURES / "user_template.json"))
    msgs = build_messages(conv, "S", tpl, "nothink", "single")
    untrained = {k + 2 for k in error_turns(conv, how)}
    segs = spans(mini_renderer, msgs, tools, "nothink", untrained)
    assert sum(flag for _, flag in segs) == trained
    assert "".join(t for t, _ in segs) == "".join(t for t, _ in spans(mini_renderer, msgs, tools, "nothink"))
    masked = [t for i, (t, flag) in enumerate(segs) if i % 2 == 1 and not flag]
    assert len(masked) == 7 - trained and all("<|tool_call>call:" in t for t in masked)
    (w, n), = windows(segs, mock_encode, max_len=100_000)
    assert n == trained
    assert [text for text, _ in w] == [text for text, _ in segs[:-1]]            # all but the closing context
    assert [flag for _, flag in w] == [flag for _, flag in segs[:-1]]


def test_mask_off_renders_as_before(converted, mini_renderer, tools):
    """off leaves every turn trained; test_windows_split_with_context_turns pins the unmasked windows."""
    conv = converted[0][0]
    tpl = load_user_template(str(FIXTURES / "user_template.json"))
    msgs = build_messages(conv, "S", tpl, "nothink", "single")
    plain = spans(mini_renderer, msgs, tools, "nothink")
    assert spans(mini_renderer, msgs, tools, "nothink", {k + 2 for k in error_turns(conv, "off")}) == plain
    assert error_turns(conv, "tool") == error_turns(conv, "all") == set()      # the fixture has no failing call


def test_window_without_trained_turn_is_dropped():
    """header 10 tokens; units of 10, 10 and 5 tokens; the first two targets are masked; max_len 30."""
    segs = [("h " * 10, 0), ("t1 " * 5, 0), ("c1 " * 5, 0), ("t2 " * 5, 0), ("c2 " * 5, 0), ("t3 " * 5, 1)]
    out = windows(segs, mock_encode, max_len=30)
    assert [[flag for _, flag in w] for w, _ in out] == [[0, 0, 0, 1]]
    assert [n for _, n in out] == [1]


def test_mask_option_choices(tmp_path):
    with pytest.raises(SystemExit):
        main([str(tmp_path / "conv.jsonl"), str(tmp_path / "out"), "--system-prompt", "s", "--user-template", "u",
              "--tokenizer", "t", "--tools", "x", "--mask-error-turns", "some"])
