"""The scorer's thinking budget: vLLM's ThinkingTokenBudgetLogitsProcessor, reproduced.

Since the 2026-09-30 harness update the scorer starts vLLM with ``--reasoning-config`` (``<|channel>`` /
``<channel|>``) and forwards an agent's ``thinking_budget`` as the request field ``thinking_token_budget``.
vLLM then counts the tokens generated after ``<|channel>``; once the count reaches the budget it forces
``<channel|>``, so the model must leave its thought and answer. The ``thought\\n`` channel label counts toward
the budget (two tokens for Gemma 4), so a budget of 24 leaves about 22 visible thought tokens.

``ThinkingBudget`` is the pure state machine (feed it tokens, ask which token to force). ``mlx_processor``
wraps it as an mlx-lm logits processor. ``complete_with_budget`` emulates it for backends without logits
processors (Ollama raw mode) by stopping at the budget, appending ``<channel|>`` and continuing.

One deliberate difference from vLLM: when a prompt ends inside an open thought, vLLM lets up to ``budget``
more tokens through before its first check; this class forces the end as soon as the budget is spent. Gemma 4
prompts built by the chat template never end inside an open thought.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

CHANNEL_START_ID = 100   # <|channel> in Gemma 4's vocabulary
CHANNEL_END_ID = 101     # <channel|>


def _last_index(seq: Sequence[int], sub: Sequence[int]) -> int:
    if not sub:
        return -1
    for i in range(len(seq) - len(sub), -1, -1):
        if list(seq[i:i + len(sub)]) == list(sub):
            return i
    return -1


@dataclass
class ThinkingBudget:
    budget: int
    start_ids: Sequence[int] = (CHANNEL_START_ID,)
    end_ids: Sequence[int] = (CHANNEL_END_ID,)
    prompt_ids: Sequence[int] | None = None
    in_think: bool = field(default=False, init=False)
    think_count: int = field(default=0, init=False)
    in_end: bool = field(default=False, init=False)
    end_count: int = field(default=0, init=False)
    forced_count: int = field(default=0, init=False)
    output: list[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.prompt_ids is not None:
            last_start = _last_index(self.prompt_ids, self.start_ids)
            last_end = _last_index(self.prompt_ids, self.end_ids)
            self.in_think = last_start > last_end
            if self.in_think:
                self.think_count = len(self.prompt_ids) - (last_start + len(self.start_ids))
        self._maybe_end()

    def _maybe_end(self) -> None:
        if self.in_think and self.think_count >= self.budget:
            self.in_think, self.in_end, self.end_count = False, True, 0

    def forced_token(self) -> int | None:
        """The token the next step must emit, or None when sampling is free."""
        return self.end_ids[self.end_count] if self.in_end else None

    def observe(self, token: int) -> None:
        """Record one generated token."""
        self.output.append(int(token))
        if self.in_end:
            self.forced_count += 1
            self.end_count += 1
            if self.end_count >= len(self.end_ids):
                self.in_end, self.end_count = False, 0
            return
        if self.output[-len(self.start_ids):] == list(self.start_ids):
            self.in_think, self.think_count = True, 0
        elif self.output[-len(self.end_ids):] == list(self.end_ids):
            self.in_think, self.think_count = False, 0
        elif self.in_think:
            self.think_count += 1
        self._maybe_end()


def apply_budget(budget: int, proposals: Sequence[int], *, start_ids: Sequence[int] = (CHANNEL_START_ID,),
                 end_ids: Sequence[int] = (CHANNEL_END_ID,), max_tokens: int | None = None) -> list[int]:
    """Emitted token stream when a model that would produce ``proposals`` runs under the budget.

    A forced token replaces the model's proposal at that step; the model then continues with its next
    proposal (a stand-in for re-sampling after the forced token).
    """
    state = ThinkingBudget(budget, start_ids, end_ids)
    out: list[int] = []
    for tok in proposals:
        if max_tokens is not None and len(out) >= max_tokens:
            break
        forced = state.forced_token()
        emit = forced if forced is not None else tok
        state.observe(emit)
        out.append(emit)
    return out


def mlx_processor(state: ThinkingBudget) -> Callable:
    """mlx-lm logits processor. mlx-lm calls it as ``proc(tokens, logits)`` where ``tokens`` holds the prompt
    tokens it was given followed by every generated token; the first call carries only prompt tokens."""
    import mlx.core as mx

    seen: list[int | None] = [None]

    def proc(tokens, logits):
        n = int(tokens.size) if hasattr(tokens, "size") else len(tokens)
        if seen[0] is None:
            seen[0] = n
        elif n > seen[0]:
            for tok in tokens[seen[0]:].tolist():
                state.observe(int(tok))
            seen[0] = n
        forced = state.forced_token()
        if forced is None:
            return logits
        logits = logits.astype(mx.float32)   # 1e9 overflows float16
        hit = mx.arange(logits.shape[-1]) == forced
        return mx.where(hit, mx.array(1e9, dtype=mx.float32), logits)

    return proc


Completion = Callable[[str, int], tuple[str, int, str]]


def complete_with_budget(complete: Completion, prompt: str, max_tokens: int, budget: int | None,
                         start: str = "<|channel>", end: str = "<channel|>") -> tuple[str, int, str, bool]:
    """Thinking budget for a text-in/text-out backend.

    ``complete(prompt, max_tokens) -> (text, completion_tokens, finish_reason)``. Generates at most
    ``budget + 1`` tokens first (the ``<|channel>`` token plus the budget); if a thought that opened at the start
    of the output is still open, appends ``<channel|>`` (counted as one generated token, as vLLM counts the forced
    token) and continues from the extended prompt. Returns (text, tokens, finish_reason, forced).
    """
    if budget is None or budget < 0:
        text, n, finish = complete(prompt, max_tokens)
        return text, n, finish, False
    text, n, finish = complete(prompt, min(max_tokens, budget + 1))
    if finish != "length" or n >= max_tokens:
        return text, n, finish, False
    forced = text.startswith(start) and end not in text
    if forced:
        text, n = text + end, n + 1
        if n >= max_tokens:
            return text, n, "length", True
    more, m, finish = complete(prompt + text, max_tokens - n)
    return text + more, n + m, finish, forced
