"""Locate the two third-party files the kit needs but does not ship.

* ``chat_template.jinja``: Gemma 4's chat template, from the model repository.
* ``gemma4_tool_parser.py``: vLLM 0.19.1's gemma4 tool parser (``vllm/tool_parsers/gemma4_tool_parser.py``
  inside the vLLM wheel). Only its pure-Python argument parser is used.

Lookup order for each file: explicit path (CLI flag) > environment variable > the kit's asset directory
(``$G4KIT_HOME`` or ``~/.cache/gemma4-swe-kit``) > for the parser only, an installed ``vllm`` package
(located without importing it). ``g4kit-assets check`` prints what was found and whether the file matches a
known fingerprint.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import shutil
import sys
import zipfile
from pathlib import Path

TEMPLATE_ENV = "G4KIT_CHAT_TEMPLATE"
PARSER_ENV = "G4KIT_TOOL_PARSER"
HOME_ENV = "G4KIT_HOME"
TEMPLATE_NAME = "chat_template.jinja"
PARSER_NAME = "gemma4_tool_parser.py"
WHEEL_PARSER_PATH = "vllm/tool_parsers/gemma4_tool_parser.py"

# sha256 fingerprints of files whose provenance was checked by hand (as of 2026-10-01)
KNOWN_TEMPLATES = {
    "94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413":
        "original Gemma 4 template (the competition's model files; used by the scorer as of 2026-10-01)",
    "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4":
        "refreshed upstream template (July 2026); NOT the one the scorer uses as of 2026-10-01",
}
KNOWN_PARSERS = {
    "682b2152b76c031df9c58c3ffbb5b243945be5d8957ebdce00faef1b9de6b889":
        "vllm 0.19.1 vllm/tool_parsers/gemma4_tool_parser.py",
}

TEMPLATE_HELP = (
    "Gemma 4 chat template not found. Download chat_template.jinja from the model repository "
    "(for example: hf download google/gemma-4-31B-it-qat-w4a16-ct chat_template.jinja --revision e3dacad5f03b852209f5ce18e44094fc80120037 --local-dir .) or copy it "
    "from the competition's model files, then either pass --template PATH, set G4KIT_CHAT_TEMPLATE, or run "
    "'g4kit-assets install-template PATH'. Check the fingerprint with 'g4kit-assets check': the scorer uses the "
    "original template, not the July 2026 refresh."
)


def asset_dir() -> Path:
    return Path(os.environ.get(HOME_ENV) or Path.home() / ".cache" / "gemma4-swe-kit")


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(path: str | Path, known: dict[str, str]) -> str:
    digest = sha256(path)
    return f"{digest[:16]}... {known.get(digest, 'unknown fingerprint')}"


def find_template(path: str | Path | None = None) -> Path | None:
    for cand in (path, os.environ.get(TEMPLATE_ENV), asset_dir() / TEMPLATE_NAME):
        if cand and Path(cand).expanduser().is_file():
            return Path(cand).expanduser()
    return None


def resolve_template(path: str | Path | None = None) -> Path:
    found = find_template(path)
    if found is None:
        raise FileNotFoundError(TEMPLATE_HELP)
    return found


def installed_vllm_parser() -> Path | None:
    """Path of an installed vLLM's gemma4 parser, found without importing vllm."""
    try:
        spec = importlib.util.find_spec("vllm")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    for root in spec.submodule_search_locations:
        cand = Path(root) / "tool_parsers" / PARSER_NAME
        if cand.is_file():
            return cand
    return None


def find_parser(path: str | Path | None = None) -> Path | None:
    for cand in (path, os.environ.get(PARSER_ENV), asset_dir() / PARSER_NAME):
        if cand and Path(cand).expanduser().is_file():
            return Path(cand).expanduser()
    return installed_vllm_parser()


def extract_parser_from_wheel(wheel: str | Path, out_dir: str | Path) -> Path:
    """Copy vllm/tool_parsers/gemma4_tool_parser.py out of a vLLM wheel."""
    out = Path(out_dir) / PARSER_NAME
    with zipfile.ZipFile(wheel) as zf:
        try:
            data = zf.read(WHEEL_PARSER_PATH)
        except KeyError:
            raise FileNotFoundError(f"{WHEEL_PARSER_PATH} not found in {wheel}") from None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return out


def install_template(src: str | Path, out_dir: str | Path) -> Path:
    out = Path(out_dir) / TEMPLATE_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="g4kit-assets", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="show which template and parser files will be used")
    c.add_argument("--template")
    c.add_argument("--parser")
    t = sub.add_parser("install-template", help="copy a chat_template.jinja into the asset directory")
    t.add_argument("path")
    t.add_argument("--dir", default=None, help="asset directory (default $G4KIT_HOME or ~/.cache/gemma4-swe-kit)")
    w = sub.add_parser("extract-parser", help="extract gemma4_tool_parser.py from a vLLM wheel")
    w.add_argument("wheel", help="e.g. vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64.whl (pip download vllm==0.19.1 --no-deps)")
    w.add_argument("--dir", default=None)
    a = ap.parse_args(argv)

    if a.cmd == "check":
        tpl, par = find_template(a.template), find_parser(a.parser)
        print(f"asset dir : {asset_dir()}")
        print(f"template  : {tpl or 'NOT FOUND'}" + (f"\n            {describe(tpl, KNOWN_TEMPLATES)}" if tpl else ""))
        print(f"parser    : {par or 'not found (the built-in parser will be used)'}"
              + (f"\n            {describe(par, KNOWN_PARSERS)}" if par else ""))
        if tpl is None:
            print("\n" + TEMPLATE_HELP, file=sys.stderr)
            return 1
        return 0
    if a.cmd == "install-template":
        out = install_template(a.path, a.dir or asset_dir())
        print(f"installed {out}\n  {describe(out, KNOWN_TEMPLATES)}")
        return 0
    out = extract_parser_from_wheel(a.wheel, a.dir or asset_dir())
    print(f"extracted {out}\n  {describe(out, KNOWN_PARSERS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
