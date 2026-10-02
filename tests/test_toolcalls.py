import json
import random

import pytest
from conftest import OFFICIAL_PARSER

from gemma4_swe_kit.toolcalls import (STRING_DELIM as D, ParserHang, builtin_parse_args, classify_calls,
                                      extract_reasoning, load_vllm_args_parser, parse_completion)

DECLARED = {"run_command", "read_file", "edit_file", "submit_patch"}


def parse(raw, **kw):
    return parse_completion(raw, parse_args=builtin_parse_args, **kw)


@pytest.mark.parametrize("text,expected", [
    ("", {}),
    (f"command:{D}ls -la{D}", {"command": "ls -la"}),
    (f"filepath:{D}a.py{D},start_line:3,end_line:10", {"filepath": "a.py", "start_line": 3, "end_line": 10}),
    ("a:true,b:false,c:null,d:None,e:1.5,f:-2,g:1e5", {"a": True, "b": False, "c": None, "d": None, "e": 1.5,
                                                       "f": -2, "g": "1e5"}),
    (f"o:{{k:{D}v{{}}{D},n:2}},a:[{D}x]{D},3]", {"o": {"k": "v{}", "n": 2}, "a": ["x]", 3]}),
    (f"u:{D}unterminated", {"u": "unterminated"}),
    (f"skill_name={D}repo-tools{D}", {}),      # kwarg syntax: no ':' before the value, so nothing is parsed
    ("k:", {"k": ""}),
])
def test_builtin_parser(text, expected):
    assert builtin_parse_args(text) == expected


def test_parser_hang_input_detected():
    with pytest.raises(ParserHang):
        builtin_parse_args(f"x:[btrue{D}-1]1{D}:\n{{")


def test_well_formed_call():
    msg, finish = parse(f"<|tool_call>call:read_file{{filepath:{D}a.py{D},start_line:2}}<tool_call|>")
    assert finish == "tool_calls" and msg["content"] is None
    (call,) = msg["tool_calls"]
    assert call["function"]["name"] == "read_file"
    assert json.loads(call["function"]["arguments"]) == {"filepath": "a.py", "start_line": 2}


def test_shell_command_as_tool_name_stays_in_content():
    """What the scorer's harness sees: no tool call, '<|tool_call>' in the content -> 'token limit' nudge."""
    raw = '<|tool_call>call:grep -rn "widen" . | head -40<tool_call|>'
    msg, finish = parse(raw)
    assert msg["tool_calls"] == [] and finish == "stop"
    assert msg["content"] == raw
    info = classify_calls(raw, DECLARED)
    assert info["unparsed"] and info["malformed_names"] == ['grep -rn "widen" . | head -40']


def test_unknown_but_well_formed_name_is_extracted():
    raw = f"<|tool_call>call:grep{{pattern:{D}widen{D}}}<tool_call|>"
    msg, finish = parse(raw)
    assert [c["function"]["name"] for c in msg["tool_calls"]] == ["grep"] and finish == "tool_calls"
    assert classify_calls(raw, DECLARED)["unknown_tools"] == ["grep"]


def test_kwarg_style_arguments_parse_to_empty_dict():
    msg, _ = parse('<|tool_call>call:run_command{command="ls"}<tool_call|>')
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {}


def test_reasoning_split_like_vllm():
    assert extract_reasoning("plain answer") == (None, "plain answer")
    assert extract_reasoning("<|channel>thought\nplan<channel|>answer") == ("plan", "answer")
    assert extract_reasoning("<|channel>thought\nstill thinking") == ("still thinking", None)
    assert extract_reasoning("lost<|channel>thought\nx<channel|>") == ("x", None)     # text before the channel is dropped
    msg, finish = parse("<|channel>thought\nlook<channel|><|tool_call>call:submit_patch{}<tool_call|>",
                        finish_reason="stop")
    assert msg["reasoning"] == "look" and msg["tool_calls"][0]["function"]["name"] == "submit_patch"
    msg, finish = parse("<|channel>thought\nnever closes", finish_reason="length")
    assert msg["content"] is None and msg["reasoning"] == "never closes" and finish == "length"


def test_text_before_call_and_trailing_stop_token():
    msg, _ = parse(f"Checking.\n<|tool_call>call:run_command{{command:{D}ls{D}}}<tool_call|><|tool_response>")
    assert msg["content"] == "Checking." and len(msg["tool_calls"]) == 1
    msg, _ = parse("All done.<turn|>")
    assert msg["content"] == "All done." and msg["tool_calls"] == []


def test_tool_choice_none_and_no_tools():
    raw = f"<|tool_call>call:run_command{{command:{D}ls{D}}}<tool_call|>"
    msg, finish = parse(raw, tool_choice="none")
    assert msg["content"] == raw and msg["tool_calls"] == []
    msg, finish = parse(raw, tools_present=False)      # vLLM parses, then drops the calls
    assert msg["tool_calls"] == [] and msg["content"] is None and finish == "stop"


def test_damaged_arguments_flagged():
    raw = f"<|tool_call>call:edit_file{{new_string:{D}x = 1`,old_string: `y{D}}}<tool_call|>"
    assert classify_calls(raw, DECLARED)["damaged_args"]


@pytest.mark.skipif(not OFFICIAL_PARSER, reason="set G4KIT_TEST_OFFICIAL_PARSER to vLLM's gemma4_tool_parser.py")
def test_builtin_parser_matches_vllm():
    official = load_vllm_args_parser(OFFICIAL_PARSER)
    alphabet = ["a", "b", ":", ",", "{", "}", "[", "]", D, " ", "1", ".", "true", "null", "\n", "x", "-"]
    rnd = random.Random(7)
    checked = 0
    for _ in range(5000):
        s = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 20)))
        try:
            expected = builtin_parse_args(s)
        except ParserHang:
            with pytest.raises(ParserHang):
                official(s)
            continue
        assert official(s) == expected, s
        checked += 1
    assert checked > 4000
