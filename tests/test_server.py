"""End-to-end proxy tests over HTTP with a scripted backend and a mock tokenizer (no model, no GPU)."""
import json

import pytest
from conftest import ServerThread, post_json, whitespace_counter

from gemma4_swe_kit.proxy.backends import Backend, GenerationResult, OllamaBackend, SamplingParams
from gemma4_swe_kit.proxy.server import ProxyApp, serve
from gemma4_swe_kit.toolcalls import STRING_DELIM as D, builtin_parse_args


class ScriptedBackend(Backend):
    name = "scripted"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, prompt, params, *, model=None, role=""):
        self.calls.append((prompt, params, model, role))
        text = self.outputs.pop(0)
        return GenerationResult(text, 0, len(text.split()), "stop")


def body(tools, **extra):
    b = {"model": "gemma-4-31b-it-qat-w4a16-ct", "tools": tools, "max_tokens": 64,
         "messages": [{"role": "system", "content": "Fix bugs."}, {"role": "user", "content": "Issue: widen()"}]}
    b.update(extra)
    return b


@pytest.fixture
def make_app(mini_renderer, tmp_path):
    def make(outputs, **kw):
        backend = ScriptedBackend(outputs)
        app = ProxyApp(mini_renderer, backend, parse_args=builtin_parse_args, token_counter=whitespace_counter,
                       log_path=str(tmp_path / "proxy.jsonl"), **kw)
        return app, backend
    return make


def test_tool_call_over_http(make_app, tools, tmp_path):
    app, backend = make_app([f"<|tool_call>call:read_file{{filepath:{D}widgets/core.py{D}}}<tool_call|>"])
    with ServerThread(serve(app, port=0)) as srv:
        status, resp = post_json(srv.url + "/v1/chat/completions", body(tools, temperature=0.6, top_k=40))
    assert status == 200
    choice = resp["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {"filepath": "widgets/core.py"}
    prompt, params, model, role = backend.calls[0]
    assert prompt.endswith("<|turn>model\n")                       # thinking on by default
    assert (params.temperature, params.top_k, params.top_p, params.max_tokens) == (0.6, 40, 0.95, 64)
    assert role == "Fix bugs."
    full = [json.loads(line) for line in open(tmp_path / "proxy_full.jsonl")]
    assert full[0]["raw"].startswith("<|tool_call>") and full[0]["body"]["max_tokens"] == 64
    summary = [json.loads(line) for line in open(tmp_path / "proxy.jsonl")]
    assert summary[0]["tool_calls"] == ["read_file"] and summary[0]["usage"]["prompt_tokens"] > 0


def test_malformed_call_reaches_client_as_content(make_app, tools, tmp_path):
    raw = '<|tool_call>call:grep -rn "widen" .<tool_call|>'
    app, _ = make_app([raw])
    status, resp, _ = app.handle(body(tools))
    assert status == 200
    msg = resp["choices"][0]["message"]
    assert msg["tool_calls"] == [] and msg["content"] == raw and resp["choices"][0]["finish_reason"] == "stop"


def test_context_overflow_returns_vllm_400(make_app, tools, tmp_path):
    app, backend = make_app(["unused"], max_model_len=40)
    with ServerThread(serve(app, port=0)) as srv:
        status, resp = post_json(srv.url + "/v1/chat/completions", body(tools, max_tokens=32))
    assert status == 400 and backend.calls == []
    assert resp["error"]["type"] == "BadRequestError" and resp["error"]["param"] == "input_tokens"
    assert resp["error"]["message"].startswith("This model's maximum context length is 40 tokens.")
    summary = [json.loads(line) for line in open(tmp_path / "proxy.jsonl")]
    assert summary[0]["status"] == 400 and summary[0]["error"] == "context_overflow"


def test_default_output_cap_is_remaining_context(make_app, tools):
    app, backend = make_app(["done"], max_model_len=1000)
    b = body(tools)
    del b["max_tokens"]
    status, _, _ = app.handle(b)
    prompt = backend.calls[0][0]
    assert status == 200 and backend.calls[0][1].max_tokens == 1000 - whitespace_counter(prompt)


def test_parser_hang_output_is_a_500(make_app, tools):
    app, _ = make_app([f"<|tool_call>call:run_command{{x:[btrue{D}-1]1{D}:\n{{}}<tool_call|>"])
    status, resp, extras = app.handle(body(tools))
    assert status == 500 and extras["error"] == "parser_hang"


def test_return_raw_and_models_endpoint(make_app, tools):
    app, _ = make_app(["<|channel>thought\nok<channel|>Done."])
    status, resp, _ = app.handle(body(tools, g4kit_return_raw=True))
    assert resp["g4kit"]["raw"].endswith("Done.") and resp["choices"][0]["message"]["reasoning"] == "ok"
    assert app.models()["data"][0]["id"] == "gemma-4-31b-it-qat-w4a16-ct"


def test_ollama_backend_request_and_budget_continuation(monkeypatch):
    """Raw-mode payload, stop tokens, and the thinking budget emulated by continuation."""
    sent = []
    replies = [{"response": "<|channel>thought\nabc", "eval_count": 4, "done_reason": "length", "prompt_eval_count": 9},
               {"response": "<|tool_call>call:submit_patch{}<tool_call|>", "eval_count": 6, "done_reason": "stop"}]

    def fake_post(self, payload):
        sent.append(payload)
        return replies.pop(0)

    monkeypatch.setattr(OllamaBackend, "_post", fake_post)
    backend = OllamaBackend(model="gemma4:31b-it-qat")
    res = backend.generate("<bos>PROMPT<|turn>model\n", SamplingParams(max_tokens=100, thinking_budget=3))
    assert sent[0]["raw"] is True and sent[0]["options"]["num_predict"] == 4
    assert "<|tool_response>" in sent[0]["options"]["stop"]
    assert sent[1]["prompt"].endswith("<|channel>thought\nabc<channel|>")
    assert sent[1]["options"]["num_predict"] == 100 - 5
    assert res.text == "<|channel>thought\nabc<channel|><|tool_call>call:submit_patch{}<tool_call|>"
    assert res.completion_tokens == 11 and res.budget_forced and res.finish_reason == "stop"
