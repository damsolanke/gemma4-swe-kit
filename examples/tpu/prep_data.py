"""Build the dataset folder that train.py reads; upload it as a private Kaggle dataset ('kaggle datasets create -p OUT').

  {variant}_train.jsonl / {variant}_valid.jsonl  hard links to g4kit-render-distill outputs (copies across file
                 systems), one pair per --variant NAME=RENDER_DIR. train.py uses
                   think      think mode, windows <= 12,288 tokens (primary)
                   think8k    think mode, windows <= 8,192 tokens (fallback after an out-of-memory error)
                   nothink    nothink mode, <= 12,288 (C2_MODE=nothink)
                   nothink8k  nothink mode, <= 8,192
  manifest.json  per variant: file names, line counts, sha256, the validation rows (random.Random(0) shuffle of the
                 valid lines, first 64), token/mask sha256 of the first 64 train windows and of those validation
                 windows (tokenizers encode of each segment with add_special_tokens=False, as the trainer does),
                 token-length statistics and the render's printed summary if it was saved as RENDER_DIR.log; the
                 tokenizer's sha256; the g4kit-curate counts with --curation-summary.
  dataset-metadata.json  with --dataset-id, for 'kaggle datasets create' (private unless you pass --public there);
                 --license defaults to CC-BY-4.0, the license of nebius/SWE-rebench-openhands-trajectories

  python examples/tpu/prep_data.py OUT --tokenizer tokenizer.json --dataset-id USER/gemma4-distill \\
      --variant think=data/r_think_12288 --variant think8k=data/r_think_8192 \\
      [--curation-summary data/curated.summary.json]

Pass the tokenizer.json of the Kaggle model the kernel trains: the trainer re-tokenizes with it and records any hash
mismatch in metrics.json. CPU only, no model weights.
"""
import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

N_VALID = 64
DESCRIPTION = ("Distillation windows for LoRA training of Gemma 4 31B, rendered by g4kit-render-distill from "
               "nebius/SWE-rebench-openhands-trajectories (CC-BY-4.0; teacher Qwen3-Coder-480B-A35B-Instruct, "
               "Apache 2.0).")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def encode(lines, tok):
    out = []
    for line in lines:
        segs = json.loads(line)["segments"]
        enc = tok.encode_batch([t for t, _ in segs], add_special_tokens=False)
        toks, mask = [], []
        for e, (_, f) in zip(enc, segs):
            toks += e.ids
            mask += [int(f)] * len(e.ids)
        out.append((np.asarray(toks, dtype="<i4"), np.asarray(mask, dtype=np.uint8)))
    return out


def lengths(lines, tok, chunk=400):
    res = []
    for c0 in range(0, len(lines), chunk):
        texts, owner = [], []
        for j, line in enumerate(lines[c0:c0 + chunk]):
            segs = json.loads(line)["segments"]
            texts += [t for t, _ in segs]
            owner += [j] * len(segs)
        acc = [0] * len(lines[c0:c0 + chunk])
        for o, e in zip(owner, tok.encode_batch(texts, add_special_tokens=False)):
            acc[o] += len(e.ids)
        res += acc
    return np.asarray(res)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", help="dataset folder to write")
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json of the model the kernel trains")
    ap.add_argument("--variant", action="append", required=True, metavar="NAME=RENDER_DIR",
                    help="a g4kit-render-distill output directory (train.jsonl, valid.jsonl); repeatable")
    ap.add_argument("--curation-summary", help="OUT.summary.json from g4kit-curate, recorded in the manifest")
    ap.add_argument("--source", default="", help="free-text provenance note stored in the manifest")
    ap.add_argument("--dataset-id", help="USER/SLUG; writes dataset-metadata.json")
    ap.add_argument("--license", default="CC-BY-4.0")
    ap.add_argument("--description", default=DESCRIPTION)
    a = ap.parse_args(argv)
    variants = dict(v.split("=", 1) for v in a.variant)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = Path(a.tokenizer)
    tok = Tokenizer.from_file(str(tokenizer))
    man = {"source": a.source,
           "format": "{'segments': [[text, train_bool], ...]}: encode each segment with add_special_tokens=False, "
                     "concatenate; loss on targets tokens[1:] where mask[1:] == 1",
           "tokenizer_json_sha256": sha(tokenizer.read_bytes()), "variants": {}}
    for variant, src in variants.items():
        sd = Path(src)
        files = {}
        for split in ("train", "valid"):
            name = f"{variant}_{split}.jsonl"
            dst = out / name
            if dst.exists():
                dst.unlink()
            try:
                os.link(sd / f"{split}.jsonl", dst)
            except OSError:
                shutil.copyfile(sd / f"{split}.jsonl", dst)
            raw = dst.read_bytes()
            files[split] = {"name": name, "bytes": len(raw), "sha256": sha(raw), "lines": raw.count(b"\n")}
        with open(out / files["train"]["name"], encoding="utf-8") as f:
            tlines = f.readlines()
        with open(out / files["valid"]["name"], encoding="utf-8") as f:
            vlines = f.readlines()
        idx = list(range(len(vlines)))
        random.Random(0).shuffle(idx)
        rows = idx[:N_VALID]
        lt = lengths(tlines, tok)
        log = Path(str(sd) + ".log")
        man["variants"][variant] = {
            "train": files["train"]["name"], "valid": files["valid"]["name"], "files": files,
            "render_log": log.read_text().strip() if log.is_file() else None,
            "valid_rows": rows,
            "check_train_first64": [{"row": i, "tokens_sha256": sha(t.tobytes()), "mask_sha256": sha(m.tobytes())}
                                    for i, (t, m) in enumerate(encode(tlines[:64], tok))],
            "check_valid": [{"row": r, "tokens_sha256": sha(t.tobytes()), "mask_sha256": sha(m.tobytes())}
                            for r, (t, m) in zip(rows, encode([vlines[r] for r in rows], tok))],
            "train_tokens": int(lt.sum()), "train_max_len": int(lt.max()), "train_p50_len": int(np.median(lt)),
            "train_share_le_4096": round(float((lt <= 4096).mean()), 4),
            "train_share_le_8192": round(float((lt <= 8192).mean()), 4)}
        print(variant, {k: v for k, v in man["variants"][variant].items()
                        if k not in ("files", "valid_rows", "check_train_first64", "check_valid")})
    if a.curation_summary:
        man["curation"] = json.loads(Path(a.curation_summary).read_text())["counts"]
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    if a.dataset_id:
        meta = {"title": a.dataset_id.split("/")[-1], "id": a.dataset_id, "licenses": [{"name": a.license}],
                "description": a.description}
        (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
    total = sum(p.stat().st_size for p in out.iterdir())
    print(f"wrote {out}: {len(list(out.iterdir()))} files, {total / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
