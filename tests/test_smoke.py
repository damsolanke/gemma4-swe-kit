"""The scripted fake model used for harness smoke tests."""
import json

from conftest import ServerThread, post_json

from gemma4_swe_kit.smoke import FakeLLM, serve, summarize, tool_result_encoding

MAIN_TOOLS = ["run_command", "read_file", "get_status", "submit_patch", "code_analyzer"]


def tool(name, props=None):
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": props or {}}}}


def request(tools, history):
    return {"model": "gemma-4-31b-it-qat-w4a16-ct", "tools": tools,
            "messages": [{"role": "system", "content": "You are the coder."}, {"role": "user", "content": "Fix it."}] + history}


def step(name, args, result, i):
    return [{"role": "assistant", "content": None,
             "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": result}]


def test_auto_script_main_agent_walks_every_tool():
    tools = [tool(n) for n in MAIN_TOOLS[:-1]] + [tool("code_analyzer", {"request": {"type": "string"}})]
    fake = FakeLLM()
    history, names = [], []
    for i in range(10):
        resp = fake.respond(request(tools, history))
        msg = resp["choices"][0]["message"]
        if not msg["tool_calls"]:
            assert msg["content"] == "Done: patch submitted."
            break
        call = msg["tool_calls"][0]["function"]
        names.append(call["name"])
        history += step(call["name"], json.loads(call["arguments"]), json.dumps({"status": "ok"}), i)
    assert names == ["read_file", "run_command", "code_analyzer", "get_status", "submit_patch"]


def test_sub_agent_session_and_custom_script():
    fake = FakeLLM(script={"roles": [{"name": "editor", "match": "^You are the editor",
                                      "steps": [{"tool": "read_file", "args": {"filepath": "x.py"}}, {"text": "OK"}]}]})
    sub = fake.respond(request([tool("read_file"), tool("run_command")], []))
    assert sub["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "read_file"
    body = request([tool("read_file")], [])
    body["messages"][0]["content"] = "You are the editor."
    first = fake.respond(body)["choices"][0]["message"]
    assert json.loads(first["tool_calls"][0]["function"]["arguments"]) == {"filepath": "x.py"}
    body["messages"] += step("read_file", {"filepath": "x.py"}, "{}", 0)
    assert fake.respond(body)["choices"][0]["message"]["content"] == "OK"
    assert [r["role"] for r in fake.records] == ["sub", "editor", "editor"]


def test_encoding_detection_and_summary():
    assert tool_result_encoding(json.dumps({"status": "ok", "stdout": "a\n"})) == "single"
    assert tool_result_encoding(json.dumps({"result": json.dumps({"status": "ok"})})) == "double"
    assert tool_result_encoding(json.dumps({"result": "OK"})) == "single"       # a sub-agent's text answer
    assert tool_result_encoding("plain text") == "raw"
    fake = FakeLLM()
    tools = [tool("read_file"), tool("submit_patch")]
    fake.respond(request(tools, []))
    fake.respond(request(tools, step("read_file", {}, json.dumps({"status": "ok"}), 0)))
    text = summarize(fake.records)
    assert "{'single': 1}" in text and "{'tool': 1}" in text


def test_http_round_trip(tmp_path):
    fake = FakeLLM(log_path=str(tmp_path / "fake.jsonl"))
    with ServerThread(serve(fake, port=0)) as srv:
        status, resp = post_json(srv.url + "/v1/chat/completions", request([tool("read_file"), tool("submit_patch")], []))
    assert status == 200 and resp["choices"][0]["finish_reason"] == "tool_calls"
    assert json.loads((tmp_path / "fake.jsonl").read_text().splitlines()[0])["role"] == "main"
