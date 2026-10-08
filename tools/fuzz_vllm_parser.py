"""Fuzz vLLM's gemma4 tool-argument parser against the kit's built-in parser and reproduce its infinite loops.

Random argument strings come from the generator of tests/test_toolcalls.py (seed 7). Where the built-in parser
returns, vLLM's _parse_gemma4_args must return the same dict. Every input the built-in parser flags as a hang
(ParserHang) is run through vLLM's own function in a child process; it counts as a hang when the child has not
returned after --timeout seconds. On vLLM 0.19.1's file, 60,000 draws give 74 flagged inputs, all of which hang, and
the parsers agree on the other 59,926.

    python tools/fuzz_vllm_parser.py PARSER_FILE [--draws 60000] [--seed 7] [--timeout 2] [--json]

PARSER_FILE is vllm/tool_parsers/gemma4_tool_parser.py (g4kit-assets extract-parser writes it). Exit status 0 when
the parsers agree and every flagged input hangs.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    from gemma4_swe_kit.toolcalls import STRING_DELIM, ParserHang, builtin_parse_args, load_vllm_args_parser
except ImportError:      # running from a checkout without the kit installed
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from gemma4_swe_kit.toolcalls import STRING_DELIM, ParserHang, builtin_parse_args, load_vllm_args_parser

# keep in step with test_builtin_parser_matches_vllm in tests/test_toolcalls.py
ALPHABET = ["a", "b", ":", ",", "{", "}", "[", "]", STRING_DELIM, " ", "1", ".", "true", "null", "\n", "x", "-"]

# the child loads only vLLM's pure helper functions, as toolcalls.load_vllm_args_parser does, minus the hang guard
CHILD = """
import json, sys
path = sys.argv[1]
src = open(path, encoding="utf-8").read()
start, end = src.index("def _parse_gemma4_value"), src.index("class Gemma4ToolParser")
ns = {"json": json, "STRING_DELIM": '<|"|>'}
exec(compile(src[start:end], path, "exec"), ns)
print(repr(ns["_parse_gemma4_args"](sys.stdin.buffer.read().decode("utf-8"))))
"""


def draws(n: int, seed: int = 7) -> list[str]:
    rnd = random.Random(seed)
    return ["".join(rnd.choice(ALPHABET) for _ in range(rnd.randint(0, 20))) for _ in range(n)]


def hangs_in_vllm(parser_file: str, s: str, timeout: float) -> bool:
    """True if vLLM's _parse_gemma4_args has not returned on s after timeout seconds (the child is killed)."""
    try:
        subprocess.run([sys.executable, "-c", CHILD, parser_file], input=s.encode("utf-8"), capture_output=True,
                       timeout=timeout)
    except subprocess.TimeoutExpired:
        return True
    return False


def fuzz(parser_file: str, n: int = 60000, seed: int = 7, timeout: float = 2.0, workers: int = 0) -> dict:
    official = load_vllm_args_parser(parser_file)
    flagged, agree, disagree = [], 0, []
    for s in draws(n, seed):
        try:
            expected = builtin_parse_args(s)
        except ParserHang:
            flagged.append(s)
            continue
        if official(s) == expected:
            agree += 1
        else:
            disagree.append(s)
    with ThreadPoolExecutor(max_workers=workers or min(8, os.cpu_count() or 1)) as pool:
        hung = list(pool.map(lambda s: hangs_in_vllm(parser_file, s, timeout), flagged))
    return {"draws": n, "seed": seed, "agree": agree, "disagree": len(disagree), "flagged": len(flagged),
            "hangs": sum(hung), "disagree_examples": disagree[:5],
            "returned_examples": [s for s, h in zip(flagged, hung) if not h][:5]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("parser_file", help="vLLM's vllm/tool_parsers/gemma4_tool_parser.py")
    ap.add_argument("--draws", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeout", type=float, default=2.0, help="seconds before a child counts as hung")
    ap.add_argument("--workers", type=int, default=0, help="children run at once (default: up to 8)")
    ap.add_argument("--json", action="store_true", help="print the result as one JSON line")
    a = ap.parse_args(argv)
    res = fuzz(os.path.expanduser(a.parser_file), a.draws, a.seed, a.timeout, a.workers)
    if a.json:
        print(json.dumps(res))
    else:
        print(f"{res['draws']} draws (seed {res['seed']}): vLLM returns and agrees on {res['agree']}, disagrees on "
              f"{res['disagree']}; the built-in parser flags {res['flagged']} as hangs, and vLLM's parser did not "
              f"return within {a.timeout:g} s on {res['hangs']} of them")
        for s in res["disagree_examples"] + res["returned_examples"]:
            print("  ", repr(s))
    return 0 if res["disagree"] == 0 and res["hangs"] == res["flagged"] else 1


if __name__ == "__main__":
    sys.exit(main())
