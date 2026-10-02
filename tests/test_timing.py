"""Scorer-time estimator on a synthetic run log."""
import json

import pytest

from gemma4_swe_kit import timing


def synthetic_run():
    results = [
        {"arm": "a", "id": "t1", "resolved": True, "started_at": 1000.0, "ended_at": 1100.0, "wall_s": 100.0,
         "patch_chars": 120},
        {"arm": "a", "id": "t2", "resolved": False, "started_at": 1200.0, "ended_at": 1500.0, "wall_s": 300.0,
         "patch_chars": 0},
    ]
    calls = [   # t, local seconds, completion tokens
        {"t": 1010.0, "dt": 20.0, "usage": {"completion_tokens": 51}, "enable_thinking": False},
        {"t": 1040.0, "dt": 30.0, "usage": {"completion_tokens": 102}, "enable_thinking": True},
        {"t": 1250.0, "dt": 100.0, "usage": {"completion_tokens": 255}, "enable_thinking": False},
        {"t": 1300.0, "dt": 50.0, "status": 400},                     # rejected request: not counted
        {"t": 5000.0, "dt": 10.0, "usage": {"completion_tokens": 999}},   # outside every task window
    ]
    return results, calls


def test_per_task_estimates():
    results, calls = synthetic_run()
    t1, t2 = timing.estimate_run(results, calls)
    assert (t1.n_calls, t1.n_thinking, t1.completion_tokens) == (2, 1, 153)
    assert t1.llm_est_s == pytest.approx(2 * 0.6 + 153 / 25.5)       # 7.2 s on the scorer
    assert t1.tool_s == pytest.approx(100 - 50)                       # wall minus local model time
    assert t1.est_s == pytest.approx(57.2)
    assert t2.n_calls == 1 and t2.est_s == pytest.approx(0.6 + 10 + 200)


def test_cap_and_projection():
    results, calls = synthetic_run()
    capped = timing.estimate_run(results, calls, cap_s=120)
    assert [round(e.est_s, 1) for e in capped] == [57.2, 120.0]
    hours = timing.project_hours(capped, n_tasks=120, setup_min=1.0)
    assert hours == pytest.approx(((57.2 + 120.0) / 2 + 60) * 120 / 3600)


def test_cli(tmp_path, capsys):
    results, calls = synthetic_run()
    rp, cp = tmp_path / "results_a.jsonl", tmp_path / "proxy.jsonl"
    rp.write_text("".join(json.dumps(r) + "\n" for r in results))
    cp.write_text("".join(json.dumps(c) + "\n" for c in calls))
    assert timing.main(["--results", str(rp), "--proxy-log", str(cp), "--cap-minutes", "2"]) == 0
    out = capsys.readouterr().out
    assert "t1" in out and "x120 tasks" in out and "solved 1/2" in out


def test_window_from_end_and_wall():
    lo, hi = timing.task_window({"ended_at": 500.0, "wall_s": 100.0})
    assert (lo, hi) == (398.0, 502.0)
