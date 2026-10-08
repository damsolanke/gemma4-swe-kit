"""g4kit-canary: check that the LoRA adapters a vLLM server serves actually change the model's logits.

Each captured request goes to the base model twice and to every adapter at temperature 0 with max_tokens 1 and the
top-5 logprobs of the first generated token. The second base run measures the server's numeric noise (prefix cache,
batch shapes). An adapter DIFFERS on a request when its top-1 token differs from the base's or the largest logprob
change over both top-5 lists exceeds max(0.05, 5 x noise); a token found in only one list is scored against the other
list's 5th logprob, a lower bound on its change. Adapter key checks pass an adapter that loads but changes nothing;
this catches it. A short greedy continuation of base and adapter is printed for each request.

Requests: a JSON object {label: request body}, a JSON list of bodies, or JSONL with one body or g4kit-proxy log entry
({"body": ...}) per line. Bodies are OpenAI chat requests as the agent sends them (messages, tools,
chat_template_kwargs, ...); the canary sets the model, temperature, max_tokens and logprob fields itself.
Results go to --out (per adapter and request). Exit status 0 when every adapter differs on at least one request,
1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

MIN_THRESHOLD = 0.05     # nats; a no-op adapter changes nothing (same kernels, one request at a time)
NOISE_MULT = 5.0
DROP = ("stream", "stream_options", "n", "max_completion_tokens", "seed", "logprobs", "top_logprobs",
        "return_tokens_as_token_ids", "user")


# ---------------------------------------------------------------------------------------------------------
# verdict math (pure)
# ---------------------------------------------------------------------------------------------------------

def top1(top: dict[str, float]) -> str:
    return max(top, key=top.get)


def max_dlogprob(a: dict[str, float], b: dict[str, float]) -> float:
    """Largest |logprob change| over both top-5 lists ({token: logprob}). A token in one list only counts with the
    other list's 5th logprob (its logprob there is at most that), which bounds its change from below."""
    d = 0.0
    for x, y in ((a, b), (b, a)):
        floor = min(y.values())
        for tok, lp in x.items():
            d = max(d, abs(lp - y[tok]) if tok in y else lp - floor)
    return d


def verdict(base: dict[str, float], base_repeat: dict[str, float], adapter: dict[str, float], *,
            min_threshold: float = MIN_THRESHOLD, noise_mult: float = NOISE_MULT) -> dict:
    """DIFFERS when the top-1 token changes or max |dlogprob| > max(min_threshold, noise_mult x noise), where the
    noise is the change between the two base runs."""
    noise = max_dlogprob(base, base_repeat)
    threshold = max(min_threshold, noise_mult * noise)
    d = max_dlogprob(base, adapter)
    return {"differs": top1(adapter) != top1(base) or d > threshold, "max_dlogprob": d, "noise": noise,
            "threshold": threshold}


def adapter_verdict(rows: list[dict]) -> tuple[bool, str]:
    """(differs on at least one request, reason when it does not)."""
    if any(r.get("differs") for r in rows):
        return True, ""
    return False, "requests failed" if rows and all(r.get("differs") is None for r in rows) else "logits unchanged"


# ---------------------------------------------------------------------------------------------------------
# requests and HTTP
# ---------------------------------------------------------------------------------------------------------

def load_requests(paths: list[str], limit: int = 0) -> dict[str, dict]:
    """{label: request body} from JSON or JSONL files (see the module docstring)."""
    out: dict[str, dict] = {}

    def add(label: str, body: object) -> None:
        if isinstance(body, dict) and "body" in body and "messages" not in body:
            body = body["body"]
        if not isinstance(body, dict) or not body.get("messages") or (limit and len(out) >= limit):
            return
        key, n = label, 2
        while key in out:
            key, n = f"{label}#{n}", n + 1
        out[key] = body

    for path in paths:
        p = Path(path)
        if p.suffix == ".jsonl":
            with open(p, encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    if line.strip():
                        add(f"{p.stem}:{i}", json.loads(line))
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for i, body in enumerate(data, 1):
                add(f"{p.stem}:{i}", body)
        elif "messages" in data or "body" in data:
            add(p.stem, data)
        else:
            for label, body in data.items():
                add(str(label), body)
    return out


class Client:
    def __init__(self, api_base: str, api_key: str = "local", timeout: float = 300.0):
        self.api_base, self.api_key, self.timeout = api_base.rstrip("/"), api_key, timeout

    def request(self, url: str, body: dict | None = None, timeout: float | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json",
                                                              "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:400]!r}") from None

    def chat(self, body: dict) -> dict:
        return self.request(self.api_base + "/chat/completions", body)

    def models(self) -> list[dict]:
        return self.request(self.api_base + "/models").get("data") or []


def canary_body(body: dict, model: str, max_tokens: int, logprobs: bool, token_ids: bool = True) -> dict:
    b = {k: v for k, v in body.items() if k not in DROP}
    b.update(model=model, temperature=0.0, max_tokens=max_tokens)
    if logprobs:
        b.update(logprobs=True, top_logprobs=5)
        if token_ids:
            b["return_tokens_as_token_ids"] = True
    return b


def top5(client: Client, model: str, body: dict) -> dict:
    """Top-5 logprobs of the first generated token (raw model logprobs, temperature 0)."""
    t = time.time()
    try:
        r = client.chat(canary_body(body, model, 1, True))
    except RuntimeError as e:   # a server without return_tokens_as_token_ids: token strings instead
        if "HTTP 400" not in str(e):
            raise
        r = client.chat(canary_body(body, model, 1, True, token_ids=False))
    sec = time.time() - t
    content = (r["choices"][0].get("logprobs") or {}).get("content") or []
    if not content or not content[0].get("top_logprobs"):
        raise RuntimeError(f"no logprobs in the response: {json.dumps(r)[:300]}")
    top = {d["token"]: float(d["logprob"]) for d in content[0]["top_logprobs"]}
    return {"top": top, "top1": top1(top), "sec": round(sec, 3), "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens")}


def greedy(client: Client, model: str, body: dict, n: int = 32) -> str:
    r = client.chat(canary_body(body, model, n, False))
    m = r["choices"][0].get("message") or {}
    thought = m.get("reasoning_content") or m.get("reasoning") or ""
    calls = [(c.get("function") or {}).get("name") for c in m.get("tool_calls") or []]
    return f"thought={thought[:160]!r} text={(m.get('content') or '')[:200]!r} calls={calls}"


def detok(client: Client, model: str, tok: str) -> str:
    """'token_id:N' -> 'token_id:N('text')' through vLLM's /detokenize; other tokens as they are."""
    if not tok.startswith("token_id:"):
        return repr(tok)
    try:
        r = client.request(client.api_base.rsplit("/v1", 1)[0] + "/detokenize",
                           {"model": model, "tokens": [int(tok.split(":", 1)[1])]}, timeout=60)
        return f"{tok}({r.get('prompt')!r})"
    except Exception:
        return tok


def discover(client: Client) -> tuple[str | None, list[str]]:
    """(base model, LoRA modules) from /v1/models: vLLM lists each LoRA module with its base model as parent."""
    data = client.models()
    base = next((m["id"] for m in data if not m.get("parent")), None)
    return base, [m["id"] for m in data if m.get("parent")]


def run_canary(client: Client, base_model: str, adapters: list[str], requests: dict[str, dict], *,
               budget_s: float = 1200.0, greedy_tokens: int = 32, min_threshold: float = MIN_THRESHOLD,
               noise_mult: float = NOISE_MULT, log: Callable[[str], None] = print) -> dict[str, list[dict]]:
    """{adapter: [per-request rows]}. Requests left when budget_s has passed are skipped."""
    res: dict[str, list[dict]] = {}
    t_start = time.time()
    for label, body in requests.items():
        if time.time() - t_start > budget_s:
            log(f"CANARY: {budget_s:.0f} s budget used up; [{label}] and later requests skipped")
            break
        try:
            b1 = top5(client, base_model, body)
            b2 = top5(client, base_model, body)
        except Exception as e:
            log(f"CANARY [{label}]: base request failed: {e}")
            for name in adapters:
                res.setdefault(name, []).append({"label": label, "error": f"base: {e}"[:300], "differs": None})
            continue
        if greedy_tokens:
            try:
                log(f"CANARY [{label}] greedy base: {greedy(client, base_model, body, greedy_tokens)}")
            except Exception as e:
                log(f"CANARY [{label}] greedy base failed: {e}")
        for name in adapters:
            try:
                a = top5(client, name, body)
            except Exception as e:
                log(f"CANARY {name}: adapter request FAILED [{label}]: {e}")
                res.setdefault(name, []).append({"label": label, "error": str(e)[:300], "differs": None})
                continue
            v = verdict(b1["top"], b2["top"], a["top"], min_threshold=min_threshold, noise_mult=noise_mult)
            log(f"CANARY {name}: top1 base={detok(client, base_model, b1['top1'])} "
                f"adapter={detok(client, base_model, a['top1'])}, max |dlogprob| over top5 = {v['max_dlogprob']:.4f}, "
                f"DIFFERS={v['differs']} [{label}: prompt {b1['prompt_tokens']} tok, noise floor {v['noise']:.4f}, "
                f"threshold {v['threshold']:.3f}, first-token latency base {b1['sec']} s / adapter {a['sec']} s]")
            if greedy_tokens:
                try:
                    log(f"CANARY {name} [{label}] greedy adapter: {greedy(client, name, body, greedy_tokens)}")
                except Exception as e:
                    log(f"CANARY {name} [{label}] greedy adapter failed: {e}")
            res.setdefault(name, []).append({"label": label, "differs": v["differs"],
                                             "max_dlogprob": round(v["max_dlogprob"], 5), "noise": round(v["noise"], 5),
                                             "threshold": v["threshold"], "base": b1, "base_repeat": b2, "adapter": a})
    for name in adapters:
        differs, why = adapter_verdict(res.get(name, []))
        if differs:
            log(f"CANARY VERDICT {name}: DIFFERS=True (the adapter changes the model on this server)")
        else:
            bar = "!" * 100
            for line in (bar, f"WARNING: CANARY VERDICT {name}: DIFFERS=False ({why}). Requests to this adapter may "
                              f"run as the BASE model; results with it say nothing about the adapter.", bar):
                log(line)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="g4kit-canary", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("requests", nargs="+", help="captured request bodies (.json or .jsonl, see the module docstring)")
    ap.add_argument("--api-base", default="http://127.0.0.1:8000/v1", help="vLLM's OpenAI-compatible endpoint")
    ap.add_argument("--api-key", default="local")
    ap.add_argument("--base-model", help="served base model name (default: the entry of /v1/models without a parent)")
    ap.add_argument("--adapter", action="append",
                    help="served LoRA module name (repeatable; default: every module /v1/models lists with a parent)")
    ap.add_argument("--limit", type=int, default=0, help="use at most N requests (0 = all)")
    ap.add_argument("--out", default="canary.json", help="per-adapter results")
    ap.add_argument("--budget-s", type=float, default=1200.0, help="skip the requests left after this many seconds")
    ap.add_argument("--timeout", type=float, default=300.0, help="seconds per HTTP request")
    ap.add_argument("--threshold", type=float, default=MIN_THRESHOLD, help="minimum max |dlogprob| that counts")
    ap.add_argument("--noise-mult", type=float, default=NOISE_MULT, help="threshold multiple of the base noise")
    ap.add_argument("--greedy-tokens", type=int, default=32, help="greedy continuation length to print (0 = none)")
    a = ap.parse_args(argv)
    requests = load_requests(a.requests, a.limit)
    if not requests:
        raise SystemExit("no request bodies with messages found in " + ", ".join(a.requests))
    client = Client(a.api_base, a.api_key, a.timeout)
    base, adapters = a.base_model, a.adapter
    if not base or not adapters:
        found_base, found = discover(client)
        base, adapters = base or found_base, adapters or found
    if not base:
        raise SystemExit("no base model name: pass --base-model")
    if not adapters:
        print(f"CANARY: {a.api_base} serves no LoRA modules; nothing to compare (pass --adapter NAME)")
        return 1
    print(f"CANARY: base {base}, adapters {adapters}, {len(requests)} request(s)", flush=True)
    res = run_canary(client, base, adapters, requests, budget_s=a.budget_s, greedy_tokens=a.greedy_tokens,
                     min_threshold=a.threshold, noise_mult=a.noise_mult, log=lambda s: print(s, flush=True))
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0 if all(adapter_verdict(res.get(name, []))[0] for name in adapters) else 1


if __name__ == "__main__":
    sys.exit(main())
