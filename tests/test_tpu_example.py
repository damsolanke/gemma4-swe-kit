"""examples/tpu without a TPU: the kernel build, the launcher's input discovery and, when numpy is installed, the
trainer's configuration list, learning-rate schedule and out-of-memory parsing."""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TPU = Path(__file__).parents[1] / "examples" / "tpu"
MODEL = "google/gemma-4/transformers/gemma-4-31b-it-qat-q4_0-unquantized/2"


def build(tmp_path, *extra):
    out = tmp_path / "kernel"
    cmd = [sys.executable, str(TPU / "build_kernel.py"), "--out-dir", str(out), "--kernel-id", "someone/g4-lora",
           "--dataset", "someone/g4-distill", *extra]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return out


def literal(src, name):
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
            return ast.literal_eval(node.value)
    raise KeyError(name)


def test_kernel_build(tmp_path):
    out = build(tmp_path, "C2_MODE=nothink")
    src = (out / "kernel.py").read_text()
    compile(src, "kernel.py", "exec")
    assert literal(src, "INNER_FILES") == {"train_lib.py": (TPU / "train_lib.py").read_text(),
                                           "train.py": (TPU / "train.py").read_text()}
    assert literal(src, "LOCK") == (TPU / "requirements.lock.txt").read_text()
    assert literal(src, "EXTRA_ENV") == {"C2_MODE": "nothink", "SMOKE_EXPECT_DEVICES": "8"}
    assert literal(src, "DATA_SLUG") == "g4-distill"
    meta = json.loads((out / "kernel-metadata.json").read_text())
    assert (meta["id"], meta["code_file"], meta["is_private"], meta["machine_shape"]) == (
        "someone/g4-lora", "kernel.py", True, "TpuV5E8")
    assert meta["dataset_sources"] == ["someone/g4-distill"] and meta["model_sources"] == [MODEL]
    bad = subprocess.run([sys.executable, str(TPU / "build_kernel.py"), "--out-dir", str(tmp_path / "x"),
                          "--kernel-id", "a/b", "--dataset", "a/c", "C2_MODE"], capture_output=True, text=True)
    assert bad.returncode == 2 and "KEY=VALUE" in bad.stderr


def test_launcher_finds_model_and_dataset(tmp_path):
    src = (build(tmp_path) / "kernel.py").read_text()
    head = src.split("# 1. TPU guard")[0]                     # environment report and input discovery only
    root = tmp_path / "input"
    model = root / "models" / "google" / "gemma-4" / "transformers" / "gemma-4-31b" / "2"
    data = root / "datasets" / "someone" / "g4-distill"
    other = root / "datasets" / "someone" / "another-dataset"
    for d in (model, data, other):
        d.mkdir(parents=True)
    (model / "model-00001-of-00002.safetensors").write_bytes(b"")
    (other / "manifest.json").write_text("{}")
    (data / "manifest.json").write_text("{}")
    (data / "think_train.jsonl").write_text("")
    env = {**os.environ, "SMOKE_INPUT_ROOT": str(root), "SMOKE_WORK": str(tmp_path / "work")}
    subprocess.run([sys.executable, "-c", head], capture_output=True, text=True, env=env, check=True, timeout=120)
    status = json.loads((tmp_path / "work" / "train_out" / "launcher.json").read_text())
    assert (status["model_dir"], status["data_dir"]) == (str(model), str(data))


TRAINER_UNITS = """
import json, sys
sys.path.insert(0, sys.argv[1])
import train as D
import train_lib as T
bad = []
for spec in ("a:q,o@30-59", "a:q,k@30-59@12k", "a:q,o@30-59@16k", "a:q@1-2@12k; a:o@1-2@12k"):
    try:
        D.parse_configs(spec)
        bad.append(spec)
    except ValueError:
        pass
lengths, rules = T.parse_fake_oom("2048, a, c@8192, e@step")
print(json.dumps({
    "configs": [[c["label"], c["hf_targets"], c["layers"], c["variant"], c["buckets"]]
                for c in D.parse_configs(D.DEFAULT_CONFIGS)],
    "accepted_bad": bad,
    "lr": [D.lr_at(s, 2e-4, 300, 1000) for s in (1, 30, 300, 650, 1000, 2000)],
    "oom": [T.oom_gb("the total memory required for HLO temporaries (19.62G) exceeds available HBM (15.75G)"),
            T.oom_gb("Used 17.43G of 15.75G hbm."), T.oom_gb("HLO temporaries (512.00M) exceeds"), T.oom_gb("none")],
    "fake_oom": [sorted(lengths), rules],
    "regex": T.lora_regex("30-31", ["q", "down"]),
}))
"""


def test_trainer_units(tmp_path):
    pytest.importorskip("numpy")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SMOKE_", "C2_", "FULL_"))}
    env["SMOKE_OUT_DIR"] = str(tmp_path)
    r = subprocess.run([sys.executable, "-c", TRAINER_UNITS, str(TPU)], capture_output=True, text=True, env=env,
                       check=True)
    res = json.loads(r.stdout)
    all5 = ["down_proj", "gate_proj", "o_proj", "q_proj", "up_proj"]
    qod = ["down_proj", "o_proj", "q_proj"]
    assert res["configs"] == [["a", all5, "30-59", "think", [8192, 12288]], ["b", all5, "30-59", "think8k", [8192]],
                              ["c", qod, "30-59", "think", [8192, 12288]], ["d", qod, "30-59", "think8k", [8192]],
                              ["e", ["o_proj", "q_proj"], "30-59", "think", [8192, 12288]],
                              ["f", ["o_proj", "q_proj"], "30-59", "think8k", [8192]]]
    assert res["accepted_bad"] == []
    assert res["lr"] == pytest.approx([2e-4 / 30, 2e-4, 2e-4, 1.1e-4, 2e-5, 2e-5])
    assert res["oom"] == [19.62, 17.43, 0.5, None]
    assert res["fake_oom"] == [[2048], [["a", None], ["c", 8192], ["e", "step"]]]
    assert res["regex"] == "layers/(30|31)/(attn/q_einsum|mlp/down_proj)"
