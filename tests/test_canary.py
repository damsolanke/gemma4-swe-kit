"""Logit canary: verdict math, request loading, and a run against a fake vLLM server with a real and a no-op adapter."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from conftest import ServerThread

from gemma4_swe_kit import canary

BASE = {"1": -0.1, "2": -2.5, "3": -3.0, "4": -4.0, "5": -5.0}


def test_max_dlogprob():
    assert canary.max_dlogprob(BASE, dict(BASE)) == 0.0
    shifted = dict(BASE, **{"1": -0.3})
    assert canary.max_dlogprob(BASE, shifted) == pytest.approx(0.2)
    assert canary.max_dlogprob(shifted, BASE) == canary.max_dlogprob(BASE, shifted)
    # '6' enters the adapter's list and '5' leaves it: each counts against the other list's 5th logprob
    swapped = {"1": -0.1, "2": -2.5, "3": -3.0, "4": -4.0, "6": -4.2}
    assert canary.max_dlogprob(BASE, swapped) == pytest.approx(0.8)          # -4.2 - (-5.0)
    tie = {"1": -0.1, "2": -2.5, "3": -3.0, "4": -4.0, "6": -5.0}           # tokens tied at 5th place swap
    assert canary.max_dlogprob(BASE, tie) == 0.0


def test_verdict():
    v = canary.verdict(BASE, dict(BASE), dict(BASE, **{"1": -0.12}))
    assert not v["differs"] and v["noise"] == 0.0 and v["threshold"] == 0.05 and v["max_dlogprob"] == pytest.approx(0.02)
    assert canary.verdict(BASE, dict(BASE), dict(BASE, **{"1": -0.2}))["differs"]          # 0.1 > 0.05
    noisy = dict(BASE, **{"1": -0.12})                                                     # base noise 0.02
    v = canary.verdict(BASE, noisy, dict(BASE, **{"1": -0.18}))
    assert v["threshold"] == pytest.approx(0.1) and not v["differs"]                      # 0.08 <= 5 x 0.02
    top1_flip = {"2": -0.69, "1": -0.70, "3": -3.0, "4": -4.0, "5": -5.0}
    assert canary.verdict(BASE, dict(BASE), top1_flip, min_threshold=10.0)["differs"]     # top-1 change alone counts
    assert canary.adapter_verdict([{"differs": False}, {"differs": True}]) == (True, "")
    assert canary.adapter_verdict([{"differs": None}, {"differs": None}]) == (False, "requests failed")
    assert canary.adapter_verdict([{"differs": None}, {"differs": False}]) == (False, "logits unchanged")


def test_request_loading(tmp_path):
    body = {"messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 8192, "stream": False}
    (tmp_path / "named.json").write_text(json.dumps({"coder": body, "analyzer": body}))
    (tmp_path / "one.json").write_text(json.dumps(body))
    (tmp_path / "many.json").write_text(json.dumps([body, {"no": "messages"}, body]))
    (tmp_path / "proxy_full.jsonl").write_text(json.dumps({"t": 0, "status": 200, "body": body}) + "\n\n"
                                               + json.dumps(body) + "\n")
    reqs = canary.load_requests([str(tmp_path / n) for n in ("named.json", "one.json", "many.json", "proxy_full.jsonl")])
    assert list(reqs) == ["coder", "analyzer", "one", "many:1", "many:3", "proxy_full:1", "proxy_full:3"]
    assert all(r == body for r in reqs.values())
    assert list(canary.load_requests([str(tmp_path / "many.json")] * 2, limit=3)) == ["many:1", "many:3", "many:1#2"]
    b = canary.canary_body(body, "adapter", 1, True)
    assert b == {"messages": body["messages"], "model": "adapter", "temperature": 0.0, "max_tokens": 1,
                 "logprobs": True, "top_logprobs": 5, "return_tokens_as_token_ids": True}


ADAPTER_TOPS = {"real": {"2": -0.2, "1": -1.9, "3": -3.0, "4": -4.0, "6": -4.5}, "noop": dict(BASE)}


class FakeVLLM(BaseHTTPRequestHandler):
    """/v1/models with two LoRA modules, /v1/chat/completions with top-5 logprobs per model, /detokenize."""

    token_ids = True             # False: answers 400 to return_tokens_as_token_ids, like a server without it
    seen: list = []

    def log_message(self, *a):
        pass

    def reply(self, code, data):
        raw = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self.reply(200, {"object": "list", "data": [{"id": "g4", "parent": None}, {"id": "real", "parent": "g4"},
                                                    {"id": "noop", "parent": "g4"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/detokenize":
            return self.reply(200, {"prompt": f"<{body['tokens'][0]}>"})
        type(self).seen.append(body)
        if body.get("return_tokens_as_token_ids") and not self.token_ids:
            return self.reply(400, {"error": "unknown field return_tokens_as_token_ids"})
        if body["model"] not in ("g4", *ADAPTER_TOPS):
            return self.reply(404, {"error": f"The model `{body['model']}` does not exist."})
        if not body.get("logprobs"):
            return self.reply(200, {"choices": [{"message": {"content": f"{body['model']} says hi", "reasoning": "plan",
                                                             "tool_calls": []}}]})
        top = ADAPTER_TOPS.get(body["model"], BASE)
        name = (lambda t: f"token_id:{t}") if body.get("return_tokens_as_token_ids") else str
        tops = [{"token": name(t), "logprob": lp} for t, lp in sorted(top.items(), key=lambda kv: -kv[1])]
        self.reply(200, {"choices": [{"message": {"content": tops[0]["token"]},
                                      "logprobs": {"content": [dict(tops[0], top_logprobs=tops)]}}],
                         "usage": {"prompt_tokens": 42, "completion_tokens": 1}})


@pytest.fixture
def server():
    FakeVLLM.seen, FakeVLLM.token_ids = [], True
    with ServerThread(ThreadingHTTPServer(("127.0.0.1", 0), FakeVLLM)) as srv:
        yield srv


@pytest.fixture
def requests_file(tmp_path):
    path = tmp_path / "requests.json"
    path.write_text(json.dumps({"coder": {"messages": [{"role": "user", "content": "fix it"}], "tools": [],
                                          "max_completion_tokens": 8192, "chat_template_kwargs": {"enable_thinking": False}}}))
    return path


def test_canary_flags_the_noop_adapter(server, requests_file, tmp_path, capsys):
    out = tmp_path / "canary.json"
    rc = canary.main([str(requests_file), "--api-base", server.url + "/v1", "--out", str(out), "--greedy-tokens", "4"])
    assert rc == 1                                                       # the no-op adapter changes nothing
    res = json.loads(out.read_text())
    assert res["real"][0]["differs"] is True and res["real"][0]["max_dlogprob"] == pytest.approx(2.3)
    assert res["noop"][0]["differs"] is False and res["noop"][0]["max_dlogprob"] == 0.0
    text = capsys.readouterr().out
    assert "CANARY: base g4, adapters ['real', 'noop']" in text
    assert "top1 base=token_id:1('<1>') adapter=token_id:2('<2>')" in text
    assert "CANARY VERDICT real: DIFFERS=True" in text and "CANARY VERDICT noop: DIFFERS=False (logits unchanged)" in text
    logprob_calls = [b for b in FakeVLLM.seen if b.get("logprobs")]
    assert [b["model"] for b in logprob_calls] == ["g4", "g4", "real", "noop"]
    assert all(b["temperature"] == 0.0 and b["max_tokens"] == 1 and b["top_logprobs"] == 5
               and "max_completion_tokens" not in b and b["chat_template_kwargs"] == {"enable_thinking": False}
               for b in logprob_calls)
    assert sorted(b["max_tokens"] for b in FakeVLLM.seen if not b.get("logprobs")) == [4, 4, 4]


def test_canary_passes_with_named_adapter_and_plain_tokens(server, requests_file, tmp_path, capsys):
    FakeVLLM.token_ids = False
    rc = canary.main([str(requests_file), "--api-base", server.url + "/v1", "--out", str(tmp_path / "c.json"),
                      "--base-model", "g4", "--adapter", "real", "--greedy-tokens", "0"])
    assert rc == 0
    assert "top1 base='1' adapter='2'" in capsys.readouterr().out
    assert not any(b.get("return_tokens_as_token_ids") for b in FakeVLLM.seen[1::2])   # each retry drops the field


def test_canary_reports_failed_requests(server, requests_file, tmp_path, capsys):
    rc = canary.main([str(requests_file), "--api-base", server.url + "/v1", "--out", str(tmp_path / "c.json"),
                      "--base-model", "g4", "--adapter", "missing"])
    assert rc == 1
    text = capsys.readouterr().out
    assert "CANARY missing: adapter request FAILED [coder]: HTTP 404" in text
    assert "CANARY VERDICT missing: DIFFERS=False (requests failed)" in text
    assert json.loads((tmp_path / "c.json").read_text())["missing"][0]["differs"] is None


def test_canary_budget_skips_the_remaining_requests(server, requests_file, tmp_path, capsys):
    rc = canary.main([str(requests_file), "--api-base", server.url + "/v1", "--out", str(tmp_path / "c.json"),
                      "--base-model", "g4", "--adapter", "real", "--budget-s", "-1"])
    assert rc == 1 and "[coder] and later requests skipped" in capsys.readouterr().out
    assert FakeVLLM.seen == []
