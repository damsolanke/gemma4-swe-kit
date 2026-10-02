"""OpenHands conversion on a hand-written trajectory, and training-window rendering with a mock tokenizer."""
import json

import pytest
from conftest import FIXTURES

from gemma4_swe_kit.chat import to_conversation
from gemma4_swe_kit.distill.convert import Conv, Skip, convert_rows, iter_rows
from gemma4_swe_kit.distill.render import build_messages, load_user_template, spans, tools_from_log, windows
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
