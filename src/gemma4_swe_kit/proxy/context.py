"""Context-length validation exactly as vLLM 0.19.1 applies it to chat completions.

vLLM rejects a request whose prompt tokens exceed ``max_model_len - max_output_tokens`` with HTTP 400, where
``max_output_tokens`` is the request's ``max_completion_tokens`` or ``max_tokens`` (0 when neither is set).
In the competition harness that exception escapes the agent loop: the task ends with "Sandbox execution
error" and an empty patch, discarding even an earlier submit_patch. Sub-agent sessions are never compacted,
so this is how long sub-agent sessions lose whole tasks.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

TokenCounter = Callable[[str], int]


class ContextOverflow(ValueError):
    """vLLM's VLLMValidationError for an over-long request (HTTP 400, BadRequestError)."""

    def __init__(self, message: str, parameter: str, value: int):
        super().__init__(message)
        self.parameter, self.value = parameter, value

    def __str__(self) -> str:   # vLLM appends the parameter and value to the message
        return f"{super().__str__()} (parameter={self.parameter}, value={self.value})"


def requested_output_tokens(body: dict) -> int | None:
    """max_completion_tokens if set, else max_tokens (None when neither is present)."""
    if body.get("max_completion_tokens") is not None:
        return int(body["max_completion_tokens"])
    if body.get("max_tokens") is not None:
        return int(body["max_tokens"])
    return None


def check_context(prompt_tokens: int, body: dict, max_model_len: int) -> None:
    """Raise ContextOverflow when vLLM 0.19.1 would reject the request."""
    if body.get("max_completion_tokens") is not None:
        out_param, out = "max_completion_tokens", int(body["max_completion_tokens"])
    else:
        out_param, out = "max_tokens", int(body.get("max_tokens") or 0)
    if out > max_model_len:
        raise ContextOverflow(f"{out_param}={out}cannot be greater than max_model_len=max_total_tokens="
                              f"{max_model_len}. Please request fewer output tokens.", out_param, out)
    max_input = max_model_len - out
    if prompt_tokens > max_input:
        qualifier = "at least " if prompt_tokens == max_input + 1 else ""
        raise ContextOverflow(
            f"This model's maximum context length is {max_model_len} tokens. However, you requested {out} output "
            f"tokens and your prompt contains {qualifier}{prompt_tokens} input tokens, for a total of "
            f"{qualifier}{prompt_tokens + out} tokens. Please reduce the length of the input prompt or the number "
            f"of requested output tokens.", "input_tokens", prompt_tokens)


def error_body(exc: Exception, status: int = 400, err_type: str = "BadRequestError") -> dict:
    """vLLM 0.19's ErrorResponse JSON."""
    param = getattr(exc, "parameter", None) if status == 400 else None
    return {"error": {"message": str(exc), "type": err_type, "param": param, "code": status}}


def bos_aware(encode: Callable[..., list[int]], bos: str = "<bos>") -> TokenCounter:
    """Count tokens of a rendered prompt; the template already starts with BOS, so none is added."""
    return lambda text: len(encode(text, add_special_tokens=not text.startswith(bos)))


def load_token_counter(spec: str) -> TokenCounter:
    """Token counter from a Hugging Face tokenizer.

    ``spec`` is a local directory or file (tokenizer.json) or a Hub repository id. transformers is used when
    installed, otherwise the lighter ``tokenizers`` package.
    """
    path = Path(os.path.expanduser(spec))
    try:
        from transformers import AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained(str(path.parent if path.is_file() else path) if path.exists() else spec)
        return bos_aware(tok.encode)
    except ImportError:
        pass
    from tokenizers import Tokenizer  # type: ignore

    if path.is_dir():
        path = path / "tokenizer.json"
    tk = Tokenizer.from_file(str(path)) if path.exists() else Tokenizer.from_pretrained(spec)
    return bos_aware(lambda text, add_special_tokens: tk.encode(text, add_special_tokens=add_special_tokens).ids)
