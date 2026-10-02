import pytest

from gemma4_swe_kit.proxy.context import ContextOverflow, check_context, error_body, requested_output_tokens


def test_fits_exactly():
    check_context(28672, {"max_tokens": 4096}, 32768)      # 28,672 + 4,096 = 32,768: allowed


def test_overflow_message_is_vllms():
    with pytest.raises(ContextOverflow) as exc:
        check_context(28700, {"max_tokens": 4096}, 32768)
    assert str(exc.value) == (
        "This model's maximum context length is 32768 tokens. However, you requested 4096 output tokens and your "
        "prompt contains 28700 input tokens, for a total of 32796 tokens. Please reduce the length of the input "
        "prompt or the number of requested output tokens. (parameter=input_tokens, value=28700)")
    body = error_body(exc.value)
    assert body == {"error": {"message": str(exc.value), "type": "BadRequestError", "param": "input_tokens",
                              "code": 400}}


def test_at_least_qualifier_at_the_boundary():
    with pytest.raises(ContextOverflow) as exc:
        check_context(28673, {"max_tokens": 4096}, 32768)
    assert "contains at least 28673 input tokens, for a total of at least 32769 tokens" in str(exc.value)


def test_max_completion_tokens_takes_precedence():
    body = {"max_completion_tokens": 2048, "max_tokens": 8192}
    assert requested_output_tokens(body) == 2048
    check_context(30720, body, 32768)
    with pytest.raises(ContextOverflow):
        check_context(30721, body, 32768)


def test_no_output_cap_checks_prompt_only():
    assert requested_output_tokens({}) is None
    check_context(32768, {}, 32768)
    with pytest.raises(ContextOverflow):
        check_context(32769, {}, 32768)


def test_output_cap_above_model_length():
    with pytest.raises(ContextOverflow) as exc:
        check_context(10, {"max_tokens": 40000}, 32768)
    assert exc.value.parameter == "max_tokens" and "cannot be greater than" in str(exc.value)
