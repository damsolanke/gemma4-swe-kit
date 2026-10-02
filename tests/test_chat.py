import json
import re

import pytest
from conftest import OFFICIAL_TEMPLATE

from gemma4_swe_kit.chat import ChatRenderer, detect_content_format, role_key, to_conversation
from gemma4_swe_kit.toolcalls import TOOL_CALL_RE, builtin_parse_args

STRING_ONLY = "{{ bos_token }}{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}{% endfor %}"
LOOP_VIA_ALIAS = ("{% set msgs = messages[1:] | list %}{% for m in msgs %}{% for p in m['content'] %}"
                  "{{ p['text'] }}{% endfor %}{% endfor %}")


def conversation(args: dict) -> list[dict]:
    return [
        {"role": "system", "content": "Fix bugs."},
        {"role": "user", "content": "Issue: widen() is wrong."},
        {"role": "assistant", "content": None, "reasoning": "read it first",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "read_file", "arguments": json.dumps(args)}}]},
        {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"status": "ok", "content": "x = 1"})},
    ]


def test_detect_content_format():
    assert detect_content_format(STRING_ONLY) == "string"
    assert detect_content_format(LOOP_VIA_ALIAS) == "openai"
    assert detect_content_format("{% for m in messages %") == "string"   # unparsable -> default


def test_mini_template_is_openai_format(mini_renderer):
    assert mini_renderer.content_format == "openai"


@pytest.mark.parametrize("args", [
    {"filepath": "widgets/core.py", "start_line": 3, "end_line": 40},
    {"command": "grep -rn \"widen\" . | head -40"},
    {"filepath": "a.py", "flags": {"raw": True, "depth": 2}, "tags": ["x", 2, False], "note": None},
    {"content": "line one\nline two\n\ttabbed {braces} [brackets]"},
])
def test_tool_call_round_trip(mini_renderer, tools, args):
    """Arguments rendered into the prompt parse back to the same dict with the vLLM-compatible parser."""
    prompt = mini_renderer.render(conversation(args), tools)
    calls = TOOL_CALL_RE.findall(prompt)
    assert [name for name, _ in calls] == ["read_file"]
    assert builtin_parse_args(calls[0][1]) == args


def test_system_turn_trailing_space_matches_vllm_openai_format(mini_renderer, tools):
    """vLLM passes content as text parts for templates that loop over content; Gemma-style system turns then
    end with a space. A plain-string renderer drops it."""
    msgs = conversation({"filepath": "a.py"})[:2]
    openai_prompt = mini_renderer.render(msgs, tools)
    string_prompt = ChatRenderer(mini_renderer.template_text, content_format="string").render(msgs, tools)
    assert "<|turn>system\n<|think|>\nFix bugs. <|tool>" in openai_prompt
    assert "<|turn>system\n<|think|>\nFix bugs.<|tool>" in string_prompt


def test_thinking_defaults_on_like_the_scorer(mini_renderer, tools):
    msgs = conversation({"filepath": "a.py"})[:2]
    on, thinking_on = mini_renderer.render_request({"messages": msgs, "tools": tools})
    off, thinking_off = mini_renderer.render_request(
        {"messages": msgs, "tools": tools, "chat_template_kwargs": {"enable_thinking": False}})
    assert thinking_on and not thinking_off
    assert "<|think|>" in on and on.endswith("<|turn>model\n")
    assert "<|think|>" not in off and off.endswith("<|turn>model\n<|channel>thought\n<channel|>")


def test_reasoning_survives_only_under_reasoning_key(mini_renderer, tools):
    """vLLM 0.19 reads 'reasoning', not 'reasoning_content', from the request history."""
    msgs = conversation({"filepath": "a.py"})
    kept = mini_renderer.render(msgs, tools)
    msgs[2] = dict(msgs[2], reasoning_content=msgs[2].pop("reasoning"))
    dropped = mini_renderer.render(msgs, tools)
    assert "<|channel>thought\nread it first\n<channel|>" in kept
    assert "read it first" not in dropped


def test_to_conversation_rules():
    conv = to_conversation([
        {"role": "system", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        {"role": "assistant", "content": "x", "tool_calls": []},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "f", "arguments": ""}},
            {"id": "2", "type": "function", "function": {"name": "g", "arguments": "{\"k\": 1}"}}]},
        {"role": "tool", "tool_call_id": "2", "content": "ok", "extra": "dropped"},
    ], "openai")
    assert conv[0]["content"] == [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert "tool_calls" not in conv[1]
    assert [c["function"]["arguments"] for c in conv[2]["tool_calls"]] == [{}, {"k": 1}]
    assert conv[3] == {"role": "tool", "content": [{"type": "text", "text": "ok"}], "tool_call_id": "2"}
    assert to_conversation([{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}],
                           "string")[0]["content"] == "a\nb"
    with pytest.raises(ValueError):
        to_conversation([{"role": "assistant", "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "f", "arguments": "{not json"}}]}])


def test_role_key():
    assert role_key([{"role": "system", "content": "You are code_editor. Edit files."}], width=19) == "You are code_editor"
    assert role_key([{"role": "user", "content": "hi"}]) == ""


@pytest.mark.skipif(not OFFICIAL_TEMPLATE, reason="set G4KIT_TEST_OFFICIAL_TEMPLATE to the official chat_template.jinja")
def test_official_template_round_trip_and_format(tools):
    renderer = ChatRenderer.from_file(OFFICIAL_TEMPLATE)
    assert renderer.content_format == "openai"
    args = {"filepath": "widgets/core.py", "start_line": 3, "end_line": 40}
    prompt = renderer.render(conversation(args), tools, chat_template_kwargs={"enable_thinking": False})
    (name, raw_args), = TOOL_CALL_RE.findall(prompt)
    assert name == "read_file" and builtin_parse_args(raw_args) == args
    assert re.search(r"Fix bugs\. <\|tool>declaration:", prompt)
