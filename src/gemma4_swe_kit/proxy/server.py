"""g4kit-proxy: an OpenAI-compatible /v1/chat/completions server that behaves like the competition scorer's
vLLM 0.19 front end, backed by Ollama (raw mode) or MLX.

Per request:
  1. render with Gemma 4's chat template after vLLM's message preprocessing (chat.ChatRenderer);
     thinking defaults to on, as the scorer's server sets ``enable_thinking: true``;
  2. reject with vLLM's HTTP 400 when prompt tokens + requested output tokens exceed max_model_len
     (needs a tokenizer: --tokenizer, or the MLX backend's own);
  3. generate in raw mode, honouring ``thinking_token_budget``;
  4. split reasoning and parse tool calls like vLLM (toolcalls.parse_completion), so malformed calls reach
     the harness exactly as they would on the scorer;
  5. log the summary, the full request body with the raw output (for replay), and suspicious outputs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .. import assets
from ..chat import ChatRenderer, role_key
from ..toolcalls import ArgsParser, ParserHang, classify_calls, declared_tool_names, load_args_parser, parse_completion
from .backends import Backend, MLXBackend, OllamaBackend, SamplingParams, parse_adapters
from .context import ContextOverflow, TokenCounter, check_context, error_body, load_token_counter, requested_output_tokens

SCORER_MODEL = "gemma-4-31b-it-qat-w4a16-ct"


class ProxyApp:
    def __init__(self, renderer: ChatRenderer, backend: Backend, *, parse_args: ArgsParser,
                 token_counter: TokenCounter | None = None, max_model_len: int = 32768,
                 served_model: str = SCORER_MODEL, adapters: list[str] | None = None, log_path: str | None = None,
                 full_log: bool = True, default_max_tokens: int = 8192):
        self.renderer, self.backend, self.parse_args = renderer, backend, parse_args
        self.count_tokens, self.max_model_len = token_counter, max_model_len
        self.served_model, self.adapters = served_model, list(adapters or [])
        self.log_path = Path(log_path) if log_path else None
        self.full_log, self.default_max_tokens = full_log, default_max_tokens
        self._log_lock = threading.Lock()

    # -- request handling -------------------------------------------------------------------------------
    def sampling(self, body: dict, n_prompt: int | None) -> SamplingParams:
        """Request sampling fields with the model's generation_config defaults (T 1.0, top_p 0.95, top_k 64)."""
        max_out = requested_output_tokens(body)
        if not max_out:
            max_out = (self.max_model_len - n_prompt) if (n_prompt is not None and self.max_model_len) \
                else self.default_max_tokens
        budget = body.get("thinking_token_budget")
        stop = body.get("stop") or []
        return SamplingParams(
            max_tokens=int(max_out),
            temperature=float(body["temperature"]) if body.get("temperature") is not None else 1.0,
            top_p=float(body["top_p"]) if body.get("top_p") is not None else 0.95,
            top_k=64 if body.get("top_k") is None else max(0, int(body["top_k"])),   # 0 = disabled
            seed=body.get("seed"),
            stop=[stop] if isinstance(stop, str) else list(stop),
            thinking_budget=int(budget) if budget is not None else None,
        )

    def handle(self, body: dict) -> tuple[int, dict, dict]:
        """Returns (HTTP status, response JSON, log record extras)."""
        if body.get("stream"):
            return 400, error_body(ValueError("stream=true is not supported by g4kit-proxy")), {}
        try:
            prompt, thinking = self.renderer.render_request(body)
        except Exception as exc:   # template errors and invalid tool-call JSON are 400s on vLLM too
            return 400, error_body(exc), {"error": f"render: {exc}"}
        n_prompt = self.count_tokens(prompt) if self.count_tokens else None
        if n_prompt is not None and self.max_model_len:
            try:
                check_context(n_prompt, body, self.max_model_len)
            except ContextOverflow as exc:
                return 400, error_body(exc), {"error": "context_overflow", "prompt_tokens": n_prompt}
        params = self.sampling(body, n_prompt)
        messages = body.get("messages") or []
        result = self.backend.generate(prompt, params, model=body.get("model"), role=role_key(messages))
        try:
            message, finish = parse_completion(result.text, parse_args=self.parse_args,
                                               finish_reason=result.finish_reason, tools_present=bool(body.get("tools")),
                                               tool_choice=body.get("tool_choice"),
                                               include_reasoning=body.get("include_reasoning", True))
        except ParserHang:
            # vLLM 0.19.1 would never answer this request; a 500 lets the client retry with a new sample
            exc = RuntimeError("vLLM 0.19.1's gemma4 tool parser loops forever on this output; on the scorer the "
                               "request would hang until the task's time budget ends")
            return 500, error_body(exc, 500, "InternalServerError"), {"error": "parser_hang", "raw": result.text}
        prompt_tokens = n_prompt if n_prompt is not None else result.prompt_tokens
        resp: dict[str, Any] = {
            "id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
            "model": body.get("model") or self.served_model,
            "choices": [{"index": 0, "message": message, "logprobs": None, "finish_reason": finish,
                         "stop_reason": None}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": result.completion_tokens,
                      "total_tokens": prompt_tokens + result.completion_tokens},
        }
        if body.get("g4kit_return_raw"):
            resp["g4kit"] = {"raw": result.text, "budget_forced": result.budget_forced}
        extras = {"raw": result.text, "enable_thinking": thinking, "budget_forced": result.budget_forced,
                  "calls": classify_calls(result.text, declared_tool_names(body.get("tools")))}
        return 200, resp, extras

    # -- logging -----------------------------------------------------------------------------------------
    def log(self, t0: float, body: dict, status: int, resp: dict, extras: dict) -> None:
        if not self.log_path:
            return
        msg = (resp.get("choices") or [{}])[0].get("message", {}) if status == 200 else {}
        raw = extras.get("raw", "")
        calls = extras.get("calls") or {}
        summary = {
            "t": round(t0, 3), "dt": round(time.time() - t0, 3), "status": status, "model": body.get("model"),
            "n_messages": len(body.get("messages") or []), "enable_thinking": extras.get("enable_thinking"),
            "thinking_token_budget": body.get("thinking_token_budget"), "budget_forced": extras.get("budget_forced"),
            "usage": resp.get("usage"), "finish": (resp.get("choices") or [{}])[0].get("finish_reason"),
            "tool_calls": [c["function"]["name"] for c in msg.get("tool_calls") or []],
            "malformed_names": calls.get("malformed_names"), "unparsed_call": calls.get("unparsed"),
            "content": (msg.get("content") or "")[:200], "raw_tail": raw[-600:],
        }
        if status != 200:
            summary["error"] = extras.get("error") or resp.get("error", {}).get("message")
            summary["prompt_tokens"] = extras.get("prompt_tokens")
        base = str(self.log_path)
        stem = base[:-6] if base.endswith(".jsonl") else base
        with self._log_lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")
            if self.full_log:
                with open(stem + "_full.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"t": round(t0, 3), "status": status, "body": body, "raw": raw,
                                        "error": summary.get("error")}, ensure_ascii=False) + "\n")
            suspicious = status == 200 and (calls.get("malformed_names") or calls.get("unparsed")
                                            or calls.get("unknown_tools") or calls.get("damaged_args")
                                            or (not msg.get("tool_calls") and not msg.get("content")))
            if suspicious:
                with open(stem + "_suspect.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"t": round(t0, 3), "body": body, "raw": raw[-4000:]}, ensure_ascii=False) + "\n")

    def models(self) -> dict:
        ids = [self.served_model] + [a for a in self.adapters if a != self.served_model]
        return {"object": "list", "data": [{"id": i, "object": "model", "owned_by": "g4kit"} for i in ids]}


def make_handler(app: ProxyApp):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def _send(self, code: int, obj: dict) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            path = self.path.split("?")[0].rstrip("/")
            if path.endswith("/models"):
                self._send(200, app.models())
            elif path in ("/health", "/v1/health", ""):
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": {"message": f"no route {self.path}", "type": "NotFoundError", "code": 404}})

        def do_POST(self) -> None:
            t0 = time.time()
            if not self.path.split("?")[0].rstrip("/").endswith("/chat/completions"):
                self._send(404, {"error": {"message": f"no route {self.path}", "type": "NotFoundError", "code": 404}})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except ValueError as exc:
                self._send(400, error_body(exc))
                return
            try:
                status, resp, extras = app.handle(body)
            except Exception as exc:   # backend failure: 500, which the harness's retry plugin retries
                status, resp, extras = 500, error_body(exc, 500, "InternalServerError"), {"error": f"{type(exc).__name__}: {exc}"}
            try:
                app.log(t0, body, status, resp, extras)
            finally:
                self._send(status, resp)

    return Handler


def serve(app: ProxyApp, host: str = "127.0.0.1", port: int = 11436) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(app))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="g4kit-proxy", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11436)
    ap.add_argument("--backend", choices=["ollama", "mlx"], default="ollama")
    ap.add_argument("--template", help="chat_template.jinja (default: $G4KIT_CHAT_TEMPLATE or the asset dir)")
    ap.add_argument("--parser", help="vLLM gemma4_tool_parser.py (default: $G4KIT_TOOL_PARSER, asset dir, installed vllm; "
                                     "falls back to the built-in parser)")
    ap.add_argument("--content-format", choices=["auto", "openai", "string"], default="auto",
                    help="message content format; auto detects it from the template like vLLM")
    ap.add_argument("--thinking-default", choices=["on", "off"], default="on",
                    help="enable_thinking when a request does not set it (the scorer's server: on)")
    ap.add_argument("--max-model-len", type=int, default=32768, help="0 disables the context check")
    ap.add_argument("--tokenizer", help="Hugging Face tokenizer dir, tokenizer.json or repo id, used to count prompt "
                                        "tokens for the context check (the MLX backend uses its own)")
    ap.add_argument("--served-model-name", default=SCORER_MODEL)
    ap.add_argument("--default-max-tokens", type=int, default=8192,
                    help="generation cap when a request sets no max_tokens and no tokenizer is available")
    ap.add_argument("--log", default=os.environ.get("G4KIT_PROXY_LOG", "g4kit_proxy.jsonl"),
                    help="summary log; *_full.jsonl (bodies + raw outputs) and *_suspect.jsonl are written next to it")
    ap.add_argument("--no-full-log", action="store_true")
    o = ap.add_argument_group("ollama backend")
    o.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    o.add_argument("--ollama-model", default="gemma4:31b-it-qat", help="any Gemma 4 31B build; raw mode bypasses its template")
    o.add_argument("--num-ctx", type=int, default=32768)
    o.add_argument("--ollama-exact-history", action="store_true",
                   help="re-insert empty thought blocks so Ollama's SWA cache is reused (faster, but changes "
                        "what the model sees; do not use for evaluation)")
    m = ap.add_argument_group("mlx backend")
    m.add_argument("--mlx-model", help="MLX model dir or repo, e.g. mlx-community/gemma-4-31B-it-qat-4bit")
    m.add_argument("--adapter", action="append", default=[], metavar="NAME=DIR",
                   help="mlx-lm LoRA adapter directory served under model name NAME (repeatable), like vLLM "
                        "--lora-modules; agents declaring 'adapter: NAME' send that model name")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    renderer = ChatRenderer.from_file(str(assets.resolve_template(a.template)), content_format=a.content_format,
                                      default_kwargs={"enable_thinking": a.thinking_default == "on"})
    parse_args, parser_src = load_args_parser(a.parser)
    adapters = parse_adapters(a.adapter)
    if a.backend == "mlx":
        if not a.mlx_model:
            raise SystemExit("--mlx-model is required with --backend mlx")
        backend: Backend = MLXBackend(a.mlx_model, adapters)
        counter = load_token_counter(a.tokenizer) if a.tokenizer else backend.token_counter()
    else:
        if adapters:
            print("note: Ollama serves the base model only; adapter names are answered by the base model", file=sys.stderr)
        backend = OllamaBackend(a.ollama_url, a.ollama_model, a.num_ctx, a.ollama_exact_history)
        counter = load_token_counter(a.tokenizer) if a.tokenizer else None
    if counter is None and a.max_model_len:
        print("warning: no tokenizer, so the max_model_len check (vLLM's 400 on context overflow) is OFF; "
              "pass --tokenizer to enable it", file=sys.stderr)
    app = ProxyApp(renderer, backend, parse_args=parse_args, token_counter=counter, max_model_len=a.max_model_len,
                   served_model=a.served_model_name, adapters=list(adapters), log_path=a.log,
                   full_log=not a.no_full_log, default_max_tokens=a.default_max_tokens)
    server = serve(app, a.host, a.port)
    tpl = assets.find_template(a.template)
    print(f"g4kit-proxy on http://{a.host}:{server.server_address[1]}/v1 | backend {backend.name} | "
          f"template {tpl} ({assets.describe(tpl, assets.KNOWN_TEMPLATES)}) | content format {renderer.content_format} | "
          f"parser {parser_src} | max_model_len {a.max_model_len} | log {a.log}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
