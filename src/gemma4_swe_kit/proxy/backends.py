"""Generation backends for the proxy. Both receive a fully rendered prompt (raw mode), so no backend-side
chat template or tool parser is involved.

* OllamaBackend: ``/api/generate`` with ``raw: true``; runs anywhere Ollama runs. The thinking budget is
  emulated by continuation (see budget.complete_with_budget).
* MLXBackend: mlx-lm on Apple Silicon, with per-role prefix KV caching (like vLLM's prefix caching), optional
  LoRA adapters selected by the request's model name (like vLLM's LoRA serving), and the exact thinking-budget
  logits processor.
"""
from __future__ import annotations

import copy
import json
import re
import threading
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from ..toolcalls import EMPTY_THOUGHT, STOP_STRINGS, TOOL_CALL_RE
from .budget import CHANNEL_END_ID, CHANNEL_START_ID, ThinkingBudget, complete_with_budget, mlx_processor


@dataclass
class SamplingParams:
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    seed: int | None = None
    stop: list[str] = field(default_factory=list)
    thinking_budget: int | None = None


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str          # "stop" or "length"
    budget_forced: bool = False


class Backend:
    name = "backend"

    def generate(self, prompt: str, params: SamplingParams, *, model: str | None = None,
                 role: str = "") -> GenerationResult:
        raise NotImplementedError

    def token_counter(self):
        return None


class OllamaBackend(Backend):
    """Ollama raw-mode generation.

    exact_history=True re-inserts the empty thought blocks that the official template drops from past model
    turns, so Ollama's sliding-window KV cache can be reused (llama.cpp cannot roll back an SWA cache, so any
    divergence forces a full re-prefill). This is much faster but changes what the model sees: it is a
    throughput option, not an evaluation setting.
    """

    name = "ollama"

    def __init__(self, url: str = "http://localhost:11434", model: str = "gemma4:31b-it-qat", num_ctx: int = 32768,
                 exact_history: bool = False, timeout: float = 3600.0):
        self.url, self.model, self.num_ctx = url.rstrip("/"), model, num_ctx
        self.exact_history, self.timeout = exact_history, timeout
        self._et_calls: dict[str, bool] = {}

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(self.url + "/api/generate", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)

    def _complete(self, prompt: str, max_tokens: int, params: SamplingParams, stats: dict) -> tuple[str, int, str]:
        # vLLM stops on generation_config's eos ids (<eos>, <turn|>, <|tool_response>); Ollama gets them as stops
        opts: dict[str, Any] = {"temperature": params.temperature, "top_p": params.top_p, "top_k": params.top_k,
                                "num_predict": max_tokens, "num_ctx": self.num_ctx,
                                "stop": list(STOP_STRINGS) + list(params.stop)}
        if params.seed is not None:
            opts["seed"] = params.seed
        r = self._post({"model": self.model, "prompt": prompt, "raw": True, "stream": False, "options": opts})
        stats.setdefault("prompt_tokens", int(r.get("prompt_eval_count") or 0))
        finish = "length" if r.get("done_reason") == "length" else "stop"
        return r.get("response", ""), int(r.get("eval_count") or 0), finish

    def _with_exact_history(self, prompt: str) -> str:
        prompt = re.sub(r"<\|turn>model\n(?!<\|channel>)", "<|turn>model\n" + EMPTY_THOUGHT, prompt)

        def restore(m: re.Match) -> str:
            return (EMPTY_THOUGHT + m.group(0)) if self._et_calls.get(m.group(0)) else m.group(0)

        return re.sub(r"(?<=<tool_response\|>)<\|tool_call>call:[\w\-\.]+\{.*?\}<tool_call\|>", restore, prompt,
                      flags=re.DOTALL)

    def generate(self, prompt: str, params: SamplingParams, *, model: str | None = None,
                 role: str = "") -> GenerationResult:
        thinking_off = prompt.endswith(EMPTY_THOUGHT)
        if self.exact_history and thinking_off:
            prompt = self._with_exact_history(prompt)
        stats: dict = {}
        text, n, finish, forced = complete_with_budget(
            lambda p, k: self._complete(p, k, params, stats), prompt, params.max_tokens, params.thinking_budget)
        if self.exact_history:
            m = TOOL_CALL_RE.search(text)
            if m:
                self._et_calls[m.group(0)] = text[: m.start()].strip() == EMPTY_THOUGHT.strip()
        return GenerationResult(text, stats.get("prompt_tokens", 0), n, finish, forced)


class MLXBackend(Backend):
    """mlx-lm generation with prefix caching.

    An agent's next request usually extends its previous prompt, so the KV cache of the previous prompt (minus
    generated tokens) is kept per (adapter, role) and only new tokens are prefilled. With thinking off the
    generation prompt ends in an empty thought block that the next request's history does not contain, so the
    snapshot stops before it.
    """

    name = "mlx"

    def __init__(self, model_path: str, adapters: dict[str, str] | None = None, prefill_step: int = 2048):
        self.model_path, self.adapters, self.prefill_step = model_path, dict(adapters or {}), prefill_step
        self._models: dict[str, Any] = {}
        self._snapshots: dict[tuple[str, str], tuple[list[int], Any]] = {}
        self._lock = threading.Lock()

    def _load(self, key: str):
        if key not in self._models:
            from mlx_lm import load

            self._models[key] = load(self.model_path, adapter_path=self.adapters.get(key))
        return self._models[key]

    def token_counter(self):
        _, tok = self._load("base")
        return lambda text: len(tok.encode(text, add_special_tokens=not text.startswith("<bos>")))

    def _prefill(self, model, tok, key: tuple[str, str], toks: list[int]):
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        snap = self._snapshots.get(key)
        if snap and len(snap[0]) < len(toks) and toks[: len(snap[0])] == snap[0]:
            cache, start = copy.deepcopy(snap[1]), len(snap[0])
        else:
            cache, start = make_prompt_cache(model), 0
        cut = len(toks) - 1
        sfx = tok.encode(EMPTY_THOUGHT, add_special_tokens=False)
        if sfx and toks[-len(sfx):] == sfx:
            cut = len(toks) - len(sfx)

        def feed(a: int, b: int) -> None:
            arr = mx.array(toks[a:b])
            for i in range(0, len(arr), self.prefill_step):
                model(arr[i:i + self.prefill_step][None], cache=cache)
                mx.eval([c.state for c in cache])

        feed(start, max(start, cut))
        self._snapshots[key] = (toks[: max(start, cut)], copy.deepcopy(cache))
        feed(max(start, cut), len(toks) - 1)
        return cache, start

    def generate(self, prompt: str, params: SamplingParams, *, model: str | None = None,
                 role: str = "") -> GenerationResult:
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        key = model if model in self.adapters else "base"
        with self._lock:
            mdl, tok = self._load(key)
            toks = tok.encode(prompt, add_special_tokens=not prompt.startswith("<bos>"))
            cache, reused = self._prefill(mdl, tok, (key, role), toks)
            if params.seed is not None:
                mx.random.seed(int(params.seed))
            sampler = make_sampler(temp=params.temperature, top_p=params.top_p, top_k=params.top_k)
            procs = []
            state = None
            if params.thinking_budget is not None and params.thinking_budget >= 0:
                start_id = tok.convert_tokens_to_ids("<|channel>") if hasattr(tok, "convert_tokens_to_ids") else CHANNEL_START_ID
                end_id = tok.convert_tokens_to_ids("<channel|>") if hasattr(tok, "convert_tokens_to_ids") else CHANNEL_END_ID
                state = ThinkingBudget(params.thinking_budget, (start_id,), (end_id,), prompt_ids=toks)
                procs.append(mlx_processor(state))
            text, n, finish = "", 0, "length"
            for resp in stream_generate(mdl, tok, toks[-1:], max_tokens=params.max_tokens, sampler=sampler,
                                        prompt_cache=cache, logits_processors=procs or None):
                text += resp.text
                n = resp.generation_tokens
                if resp.finish_reason:
                    finish = resp.finish_reason
            for stop in params.stop:
                if stop and stop in text:
                    text, finish = text[: text.index(stop)], "stop"
            forced = bool(state and state.forced_count)
            print(f"mlx {key} role={role[:24]!r} prompt {len(toks)} reused {reused} generated {n}", flush=True)
            return GenerationResult(text, len(toks), n, finish, forced)


def parse_adapters(specs: list[str] | None) -> dict[str, str]:
    """``NAME=DIR`` pairs; requests whose model field equals NAME use that LoRA adapter."""
    out = {}
    for spec in specs or []:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"--adapter expects NAME=DIR, got {spec!r}")
        out[name] = path
    return out
