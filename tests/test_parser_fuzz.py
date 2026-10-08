"""vLLM 0.19.1's gemma4 argument parser loops forever on 74 of 60,000 random argument strings (seed 7); the built-in
parser flags exactly those and agrees with vLLM on the rest. Slow (about a minute), so it runs only when
G4KIT_TEST_FUZZ_PARSER points to vLLM's gemma4_tool_parser.py."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import FUZZ_PARSER

ROOT = Path(__file__).parents[1]


@pytest.mark.skipif(not FUZZ_PARSER, reason="set G4KIT_TEST_FUZZ_PARSER to vLLM's gemma4_tool_parser.py (slow)")
def test_vllm_parser_hangs_in_60000_draws():
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")]))}
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "fuzz_vllm_parser.py"), os.path.expanduser(FUZZ_PARSER),
                        "--json"], capture_output=True, text=True, env=env, timeout=900)
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert {k: res[k] for k in ("draws", "seed", "agree", "disagree", "flagged", "hangs")} == {
        "draws": 60000, "seed": 7, "agree": 59926, "disagree": 0, "flagged": 74, "hangs": 74}
    assert r.returncode == 0
