"""The harness runner needs the competition wheelhouse; only its harness-independent parts run everywhere."""
import importlib.util

import pytest

from gemma4_swe_kit import harness


def test_cli_parses_without_the_harness():
    with pytest.raises(SystemExit) as exc:
        harness.main(["run", "--help"])
    assert exc.value.code == 0


@pytest.mark.skipif(importlib.util.find_spec("yaml") is None, reason="pyyaml not installed")
def test_eval_config_with_and_without_evaluation_key(tmp_path):
    (tmp_path / "eval_config.yaml").write_text("evaluation:\n  max_time_minutes: 8\n  max_tool_calls: 100\n")
    assert harness.load_eval_config(tmp_path) == {"max_time_minutes": 8, "max_tool_calls": 100}
    (tmp_path / "eval_config.yaml").write_text("timeout_seconds: 300\n")
    assert harness.load_eval_config(tmp_path) == {"timeout_seconds": 300}
    assert harness.load_eval_config(tmp_path / "missing") == {}


@pytest.mark.skipif(importlib.util.find_spec("swegemma") is None, reason="needs the competition harness (swegemma)")
def test_user_template_from_installed_harness():
    tpl = harness.user_template(8.0, 100, 500)
    assert harness.REPO in tpl["user_message"] and harness.PROBLEM in tpl["user_message"]
    assert harness.TREE in tpl["tree_section"] and "- Time allowance: 8.0 minutes" in tpl["user_message"]
