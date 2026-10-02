"""The thinking-budget logits processor logic, tested as a pure function on fake token streams."""
from gemma4_swe_kit.proxy.budget import ThinkingBudget, apply_budget, complete_with_budget

START, END, T, CALL = 100, 101, 7, 48     # <|channel>, <channel|>, a thought token, <|tool_call>


def thought_tokens(stream):
    """Tokens between the first <|channel> and the next <channel|>."""
    i = stream.index(START)
    return stream[i + 1: stream.index(END, i)]


def test_forces_end_after_budget():
    model = [START] + [T] * 50 + [END, CALL]          # a model that would think for 50 tokens
    out = apply_budget(5, model)
    assert out[:7] == [START, T, T, T, T, T, END]
    assert len(thought_tokens(out)) == 5


def test_short_thought_is_untouched():
    model = [START, T, T, END, CALL, CALL]
    assert apply_budget(10, model) == model


def test_budget_counts_from_each_new_thought():
    model = [START, T, T, END, CALL, START] + [T] * 9 + [END]
    out = apply_budget(3, model)
    assert out == [START, T, T, END, CALL, START, T, T, T, END, T, T, T, T, T, END]


def test_budget_zero_closes_immediately():
    assert apply_budget(0, [START, T, T, T])[:2] == [START, END]


def test_no_thought_no_forcing():
    model = [CALL, T, T, T, T, T]
    assert apply_budget(1, model) == model


def test_multi_token_end_sequence():
    out = apply_budget(2, [1, T, T, T, T, T], start_ids=(1,), end_ids=(8, 9))
    assert out == [1, T, T, 8, 9, T]


def test_state_from_prompt_inside_thought():
    state = ThinkingBudget(4, prompt_ids=[2, 105, START, T, T])   # prompt already holds 2 thought tokens
    for tok in (T, T):
        assert state.forced_token() is None
        state.observe(tok)
    assert state.forced_token() == END
    state.observe(END)
    assert state.forced_token() is None and state.forced_count == 1


def test_continuation_emulation_for_text_backends():
    """complete_with_budget: stop at budget + 1 tokens, append <channel|>, continue from the extended prompt."""
    calls = []

    def fake_complete(prompt, max_tokens):
        calls.append((prompt, max_tokens))
        if len(calls) == 1:
            return "<|channel>thought\nmany words", max_tokens, "length"
        return "<|tool_call>call:submit_patch{}<tool_call|>", 9, "stop"

    text, n, finish, forced = complete_with_budget(fake_complete, "P", 1000, 24)
    assert calls[0] == ("P", 25)
    assert calls[1] == ("P<|channel>thought\nmany words<channel|>", 1000 - 26)
    assert forced and finish == "stop" and n == 25 + 1 + 9
    assert text.endswith("<channel|><|tool_call>call:submit_patch{}<tool_call|>")


def test_continuation_skipped_when_thought_closed_or_no_budget():
    def closed(prompt, max_tokens):
        return "<|channel>thought\nok<channel|>Done.", 6, "stop"

    assert complete_with_budget(closed, "P", 100, 24) == ("<|channel>thought\nok<channel|>Done.", 6, "stop", False)
    seen = []
    complete_with_budget(lambda p, k: seen.append(k) or ("x", 1, "stop"), "P", 100, None)
    assert seen == [100]
