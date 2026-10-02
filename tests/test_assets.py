import zipfile

import pytest

from gemma4_swe_kit import assets
from gemma4_swe_kit.toolcalls import load_args_parser

FAKE_PARSER = '''import json
STRING_DELIM = '<|"|>'

def _parse_gemma4_value(value_str):
    return value_str

def _parse_gemma4_args(args_str, *, partial=False):
    return {"parsed_by": "file", "raw": args_str}

class Gemma4ToolParser:
    pass
'''


def test_extract_parser_from_wheel_and_load(tmp_path, monkeypatch):
    wheel = tmp_path / "vllm-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.writestr(assets.WHEEL_PARSER_PATH, FAKE_PARSER)
    out = assets.extract_parser_from_wheel(wheel, tmp_path / "home")
    assert out.read_text() == FAKE_PARSER
    monkeypatch.setenv(assets.HOME_ENV, str(tmp_path / "home"))
    monkeypatch.delenv(assets.PARSER_ENV, raising=False)
    monkeypatch.delenv(assets.TEMPLATE_ENV, raising=False)
    parse, source = load_args_parser()
    assert source.startswith("vllm:") and parse("k:1") == {"parsed_by": "file", "raw": "k:1"}
    assert assets.main(["check"]) == 1        # parser found, template missing


def test_builtin_parser_when_nothing_found(tmp_path, monkeypatch):
    monkeypatch.setenv(assets.HOME_ENV, str(tmp_path))
    monkeypatch.delenv(assets.PARSER_ENV, raising=False)
    monkeypatch.setattr(assets, "installed_vllm_parser", lambda: None)
    parse, source = load_args_parser()
    assert source == "builtin" and parse('command:<|"|>ls<|"|>') == {"command": "ls"}


def test_template_lookup_order(tmp_path, monkeypatch):
    home = tmp_path / "home"
    explicit = tmp_path / "explicit.jinja"
    explicit.write_text("A")
    env_file = tmp_path / "env.jinja"
    env_file.write_text("B")
    monkeypatch.setenv(assets.HOME_ENV, str(home))
    monkeypatch.delenv(assets.TEMPLATE_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match="Gemma 4 chat template not found"):
        assets.resolve_template()
    installed = assets.install_template(explicit, home)
    assert assets.resolve_template() == installed
    monkeypatch.setenv(assets.TEMPLATE_ENV, str(env_file))
    assert assets.resolve_template() == env_file
    assert assets.resolve_template(explicit) == explicit
    assert "unknown fingerprint" in assets.describe(explicit, assets.KNOWN_TEMPLATES)
