"""Assemble the Kaggle TPU kernel: launcher_template.py with train.py, train_lib.py and requirements.lock.txt embedded
(one script, kernel.py) and its kernel-metadata.json, both written to --out-dir for 'kaggle kernels push -p OUT_DIR'.

  python examples/tpu/build_kernel.py --out-dir build/tpu --kernel-id USER/gemma4-tpu-lora --dataset USER/gemma4-distill \\
      [--model google/gemma-4/transformers/gemma-4-31b-it-qat-q4_0-unquantized/2] [KEY=VALUE ...]

KEY=VALUE pairs are trainer settings (C2_*, SMOKE_*, FULL_* environment variables) baked into the kernel, e.g.
C2_MODE=nothink or C2_CONFIGS='e:q,o@30-59@12k'. The kernel is private, TPU v5e-8, internet on (the launcher installs
the lock with uv).
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "google/gemma-4/transformers/gemma-4-31b-it-qat-q4_0-unquantized/2"


def build(out_dir, kernel_id, dataset, model=MODEL, env=None, session_timeout_s=32000):
    tpl = (HERE / "launcher_template.py").read_text()
    files = {"train_lib.py": (HERE / "train_lib.py").read_text(), "train.py": (HERE / "train.py").read_text()}
    lock = (HERE / "requirements.lock.txt").read_text()
    extra = dict(env or {})
    extra.setdefault("SMOKE_EXPECT_DEVICES", "8")
    slug = dataset.split("/")[-1]
    for tok, val in (("__INNER_FILES__", repr(files)), ("__LOCK__", repr(lock)), ("__EXTRA_ENV__", repr(extra)),
                     ("__DATA_SLUG__", repr(slug))):
        assert tpl.count(tok) == 1, tok
        tpl = tpl.replace(tok, val)
    out_dir = Path(out_dir)
    compile(tpl, str(out_dir / "kernel.py"), "exec")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kernel.py").write_text(tpl)
    meta = {"id": kernel_id, "title": kernel_id.split("/")[-1], "code_file": "kernel.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": True,
            "enable_internet": True, "machine_shape": "TpuV5E8", "session_timeout_seconds": session_timeout_s,
            "dataset_sources": [dataset], "model_sources": [model], "competition_sources": [], "kernel_sources": []}
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return tpl, extra


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", required=True, help="where kernel.py and kernel-metadata.json are written")
    ap.add_argument("--kernel-id", required=True, help="Kaggle kernel id, USER/SLUG")
    ap.add_argument("--dataset", required=True, help="Kaggle dataset id of the prep_data.py folder, USER/SLUG")
    ap.add_argument("--model", default=MODEL, help="Kaggle model source with the Gemma 4 31B checkpoint")
    ap.add_argument("--session-timeout-s", type=int, default=32000, help="above the launcher's 8.5 h guard")
    ap.add_argument("env", nargs="*", metavar="KEY=VALUE", help="trainer settings baked into the kernel")
    a = ap.parse_args(argv)
    bad = [x for x in a.env if "=" not in x]
    if bad:
        ap.error(f"expected KEY=VALUE, got {bad}")
    tpl, extra = build(a.out_dir, a.kernel_id, a.dataset, a.model, dict(x.split("=", 1) for x in a.env),
                       a.session_timeout_s)
    print(f"wrote {Path(a.out_dir) / 'kernel.py'} ({len(tpl)} chars) and kernel-metadata.json, extra env {extra}")


if __name__ == "__main__":
    main()
