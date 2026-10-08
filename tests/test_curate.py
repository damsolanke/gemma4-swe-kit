"""Trajectory curation: tool-error rules, normalization, scoring and the deterministic greedy selection."""
import json

import pytest

from gemma4_swe_kit.distill import curate

FIND = "find . -maxdepth 2 -not -path '*/.*'"
PATCH_A = ("diff --git a/widgets/core.py b/widgets/core.py\n--- a/widgets/core.py\n+++ b/widgets/core.py\n"
           "@@ -1,2 +1,2 @@\n def widen(width, pad):\n-    return width + abs(pad)\n+    return width + max(pad, 0)\n")
PATCH_B = ("diff --git a/tests/test_core.py b/tests/test_core.py\n--- a/tests/test_core.py\n+++ b/tests/test_core.py\n"
           "@@ -1 +1 @@\n-assert widen(1, -1) == 2\n+assert widen(1, -1) == 1\n"
           "diff --git a/scratch.py b/scratch.py\nnew file mode 100644\n--- /dev/null\n+++ b/scratch.py\n@@ -0,0 +1 @@\n"
           "+print(1)\n")


def ok(**kw):
    return {"status": "ok", **kw}


def err(msg, **details):
    d = {"status": "error", "error_type": "ToolError", "error_message": msg}
    if details:
        d["details"] = details
    return d


def traj(instance_id, repo, steps, patch):
    """steps: [(name, args, result, reasoning)]; a submit_patch call for `patch` and the final text are appended."""
    msgs = []
    steps = steps + [("submit_patch", {}, ok(patch_size=len(patch), files_changed=patch.count("diff --git")), None)]
    for i, (name, args, result, reasoning) in enumerate(steps):
        cid = f"call_{i}"
        msgs.append({"role": "assistant", "content": None, "reasoning": reasoning,
                     "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": args}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "name": name, "content": json.dumps(result)})
    msgs.append({"role": "assistant", "content": "Submitted the fix.", "reasoning": None})
    return {"instance_id": instance_id, "repo": repo, "issue": "widen() grows on negative pad", "n_turns": len(steps),
            "messages": msgs}


def good():
    return traj("acme__widgets-1", "acme/widgets", [
        ("run_command", {"command": FIND}, ok(stdout="./widgets", stderr="", exit_code=0), "Look around."),
        ("run_command", {"command": FIND}, ok(stdout="./widgets\n./widgets/core.py", stderr="", exit_code=0), "Deeper."),
        ("read_file", {"filepath": "widgets/core.py"}, ok(content="def widen(width, pad): ..."), None),
        ("run_command", {"command": "python /tmp/repro.py"},
         err("Traceback (most recent call last):\nAssertionError", stdout="", stderr="Traceback", exit_code=1), None),
        ("edit_file", {"filepath": "widgets/core.py", "old_string": "abs(pad)", "new_string": "max(pad, 0)"}, ok(), None),
        ("run_command", {"command": "cd /workspace && python -m pytest -q tests/test_core.py"},
         ok(stdout="2 passed", stderr="", exit_code=0), None),
    ], PATCH_A)


def sloppy():
    return traj("acme__widgets-2", "acme/widgets", [
        ("read_file", {"filepath": "missing.py"}, err("File not found: missing.py"), None),
        ("read_file", {"filepath": "missing.py"}, err("File not found: missing.py"), None),
        ("run_command", {"command": "pip install widgets-extra"}, ok(stdout="", stderr="", exit_code=0), None),
        ("run_command", {"command": "grpe -rn widen ."},
         err("grpe: command not found", stdout="", stderr="grpe: command not found", exit_code=127), None),
        ("edit_file", {"filepath": "tests/test_core.py", "old_string": "2", "new_string": "1"}, ok(), "x" * 2500),
    ], PATCH_B)


@pytest.mark.parametrize("name,args,result,expected", [
    ("read_file", {"filepath": "a.py"}, err("File not found"), True),
    ("run_command", {"command": "python -m pytest -q tests"}, err("1 failed", exit_code=1), False),
    ("run_command", {"command": "pytest tests/test_x.py"}, err("ERROR: file or directory not found", exit_code=4), True),
    ("run_command", {"command": "foo --bar"}, err("foo: command not found", exit_code=127), True),
    ("run_command", {"command": "grep -rn needle ."}, err("", stdout="", stderr="", exit_code=1), False),
    ("run_command", {"command": "grep -rn needle missing/"},
     err("grep: missing/: No such file or directory", exit_code=2), True),
    ("run_command", {"command": "python /tmp/x.py"},
     err("python: can't open file '/tmp/x.py': [Errno 2] No such file or directory", exit_code=2), True),
    ("run_command", {"command": "cd /workspace && python repro.py"},
     err("Traceback (most recent call last):\nValueError", exit_code=1), False),
    ("edit_file", {"filepath": "a.py"}, ok(), False),
])
def test_tool_error_rules(name, args, result, expected):
    assert curate.tool_error(name, args, result) is expected


def test_command_classes():
    assert curate.main_program("cd /workspace && FOO=1 timeout 30 python3.12 -m pytest -q") == ("python", "-m pytest -q")
    assert curate.classify_command("sed -i 's/a/b/' widgets/core.py") == "edit"
    assert curate.classify_command("cat > /tmp/repro.py <<'EOF'\nprint(1)\nEOF") == "scratch"
    assert curate.classify_command("grep -rn widen .") == "inspect"
    assert curate.classify_command("python -m pytest -q") == "run" and curate.is_test_command("python -m pytest -q")


def test_normalize_drops_the_superseded_listing():
    conv, dropped = curate.normalize(good())
    assert dropped == 1 and conv["n_turns"] == 6
    first = conv["messages"][0]
    assert first["tool_calls"][0]["function"]["arguments"] == {"command": FIND}
    assert first["reasoning"] == "Look around.\n\nDeeper."             # the dropped turn's note moves to the kept one
    assert json.loads(conv["messages"][1]["content"])["stdout"] == "./widgets\n./widgets/core.py"
    conv, dropped = curate.normalize(sloppy())                         # a pure repeat: the later turn goes
    assert dropped == 1 and conv["n_turns"] == 5
    assert [m["tool_calls"][0]["id"] for m in conv["messages"] if m.get("tool_calls")][:2] == ["call_0", "call_2"]


def test_features_and_score():
    conv, _ = curate.normalize(good())
    score, f, raw = curate.features(conv, PATCH_A)
    assert f == {"order": 1.0, "repeat": 1.0, "errors": 1.0, "retry": 1.0, "patch": 1.0, "brevity": 1.0}
    assert score == pytest.approx(1.0)
    assert raw["read_before_edit"] == 1.0 and raw["verify_after_edit"] and raw["run_before_edit"]
    assert raw["tool_errors"] == 0 and raw["patch"]["src_changed"] == 2

    score_b, fb, rawb = curate.features(sloppy(), PATCH_B)
    assert (rawb["tool_errors"], rawb["forbidden"], rawb["failed_retries"], rawb["redundant"]) == (3, 1, 1, 1)
    assert fb["errors"] == 0.0                                          # (3 errors + 3 x 1 forbidden) / 6 calls
    assert rawb["patch"]["edits_existing_test"] and rawb["patch"]["adds_root_file"]
    assert fb["patch"] == pytest.approx(0.3 * 0.6)
    assert rawb["long_thoughts"] == 1
    assert score_b < 0.5 < score
    assert curate.features(sloppy(), None)[1]["patch"] == 0.0


def test_patch_matching_follows_source_order():
    a1, a2 = good(), good()
    rows = [("acme__widgets-1", 0, PATCH_A),                     # unresolved: never matched
            ("acme__widgets-1", 1, PATCH_A + "\n"),              # resolved, but a different size
            ("acme__widgets-1", 1, PATCH_A),
            ("acme__widgets-1", 1, PATCH_A)]
    assert curate.load_patches([a1, a2], rows) == [PATCH_A, PATCH_A]
    assert curate.load_patches([a1, a2, good()], rows) == [PATCH_A, PATCH_A, None]


def test_selection_is_greedy_with_penalties():
    items = [(0, "i1", "r1", 0.90), (1, "i1", "r1", 0.89), (2, "i2", "r1", 0.88), (3, "i3", "r2", 0.70)]
    assert curate.select(items, 3, 8, 0.02, 0.03) == [0, 2, 1]
    assert curate.select(items, 3, 2, 0.02, 0.03) == [0, 2, 3]      # repository cap
    assert curate.select(list(reversed(items)), 3, 8, 0.02, 0.03) == [0, 2, 1]
    tied = [(5, "b", "r1", 0.5), (4, "a", "r2", 0.5)]
    assert curate.select(tied, 1, 8, 0.0, 0.0) == [4]                # ties go to the lower instance id


def test_main_is_deterministic(tmp_path, capsys):
    convs = [good(), sloppy(), traj("encode__httpx-9", "encode/httpx", [], PATCH_A)]
    src = tmp_path / "conv.jsonl"
    src.write_text("".join(json.dumps(c) + "\n" for c in convs))
    patches = tmp_path / "rows.jsonl"
    patches.write_text("".join(json.dumps({"instance_id": c["instance_id"], "resolved": 1,
                                           "model_patch": PATCH_B if c is convs[1] else PATCH_A}) + "\n"
                               for c in convs))
    outputs = []
    for run in ("a", "b"):
        out = tmp_path / run / "curated.jsonl"
        out.parent.mkdir()
        assert curate.main([str(src), str(out), "--source", str(patches), "--keep", "0.5"]) == 0
        outputs.append([(out.parent / name).read_bytes()
                        for name in ("curated.jsonl", "curated.scores.jsonl", "curated.summary.json")])
    assert outputs[0] == outputs[1]
    kept = [json.loads(line) for line in outputs[0][0].decode().splitlines()]
    assert [c["instance_id"] for c in kept] == ["acme__widgets-1"]
    assert kept[0]["n_turns"] == 6                                    # normalized
    summary = json.loads(outputs[0][2])
    assert summary["counts"]["excluded_repo"] == 1 and summary["counts"]["eligible"] == 2
    assert summary["counts"]["target"] == 1 and summary["params"]["exclude_repos"] == ["encode/httpx"]
    scores = [json.loads(line) for line in outputs[0][1].decode().splitlines()]
    assert [(r["kept"], r["eligible"]) for r in scores] == [(True, True), (False, True), (False, False)]
    assert "kept 1 of 2 eligible" in capsys.readouterr().out


def test_cli_requires_the_patch_source(tmp_path):
    with pytest.raises(SystemExit):
        curate.main([str(tmp_path / "conv.jsonl"), str(tmp_path / "out.jsonl")])
