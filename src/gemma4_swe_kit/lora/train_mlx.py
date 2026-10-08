"""g4kit-train-mlx: LoRA fine-tuning of Gemma 4 31B (MLX 4-bit QAT base) on Apple Silicon, on multi-turn windows with
per-segment loss masks (g4kit-render-distill output) or on pre-rendered prompt / completion pairs; the format is
detected from the data.

DATA holds train.jsonl and valid.jsonl. Windows ({"segments": [[text, train], ...]}) are tokenized segment by segment
(segment boundaries sit on special tokens) and concatenated; the loss is on train=1 tokens. Prompt / completion pairs
({"prompt", "completion"}) are already rendered with the chat template, so none is applied here; the loss is on the
completion tokens only (the prompt is masked through mlx_lm's (tokens, offset) batches), and each completion ends with
the stop token the server stops on after a tool call (--stop-token). Adapters target q_proj and o_proj by default, the
module set of the competition's sample adapter, which loads in the scorer's vLLM.

Writes OUT/adapters.safetensors (mlx_lm format, with a checkpoint every --save-every steps) and
OUT/adapter_config.json; g4kit-mlx-to-peft converts the result for vLLM. Needs mlx-lm (pip install '.[mlx]').
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BASE = "mlx-community/gemma-4-31B-it-qat-4bit"


class PreRendered:
    """{"prompt", "completion"} pairs as (tokens, prompt length), for mlx_lm's default batching and loss."""

    def __init__(self, path, tok, stop_ids, max_len, limit=0):
        self._data, skipped = [], 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if limit and len(self._data) >= limit:
                    break
                r = json.loads(line)
                p = tok.encode(r["prompt"], add_special_tokens=False)
                c = tok.encode(r["completion"], add_special_tokens=False) + list(stop_ids)
                if len(p) + len(c) > max_len:
                    skipped += 1
                    continue
                self._data.append((p + c, len(p)))
        print(f"{path}: {len(self._data)} examples, {skipped} longer than {max_len} tokens skipped", flush=True)

    def process(self, d):
        return d

    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return len(self._data)


class Segments:
    """Multi-turn windows from g4kit-render-distill: {"segments": [[text, train], ...]}; loss on train=1 text.
    Segment boundaries sit on special tokens, so segments are tokenized separately and concatenated."""

    def __init__(self, path, tok, max_len, limit=0):
        self._data, skipped = [], 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if limit and len(self._data) >= limit:
                    break
                toks, mask = [], []
                for text, train in json.loads(line)["segments"]:
                    t = tok.encode(text, add_special_tokens=False)
                    toks += t
                    mask += [int(train)] * len(t)
                if len(toks) > max_len or not any(mask):
                    skipped += 1
                    continue
                self._data.append((toks, mask))
        print(f"{path}: {len(self._data)} windows, {sum(sum(m) for _, m in self._data)} target tokens, "
              f"{skipped} skipped (longer than {max_len} or no target)", flush=True)

    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return len(self._data)


def seg_iterate_batches(dataset, batch_size, max_seq_length, loop=False, seed=None, comm_group=None):
    """batch_size 1: (tokens [1, L], target mask [1, L]) in random order."""
    import mlx.core as mx
    import numpy as np

    idx = np.arange(len(dataset))
    while True:
        for i in np.random.permutation(idx):
            toks, mask = dataset[int(i)]
            yield mx.array([toks]), mx.array([mask])
        if not loop:
            break


def seg_loss(model, batch, mask):
    import mlx.core as mx
    import mlx.nn as nn

    logits = model(batch[:, :-1])
    m = mask[:, 1:].astype(mx.float32)
    ce = nn.losses.cross_entropy(logits, batch[:, 1:]) * m
    ntoks = m.sum()
    return ce.astype(mx.float32).sum() / ntoks, ntoks


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="g4kit-train-mlx", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", required=True, help="directory with train.jsonl and valid.jsonl")
    ap.add_argument("--out", required=True, help="adapter output directory")
    ap.add_argument("--base", default=BASE, help="MLX base model (Hugging Face repo id or local directory)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scale", type=float, default=2.0, help="LoRA scale; PEFT lora_alpha = scale x rank")
    ap.add_argument("--keys", default="self_attn.q_proj,self_attn.o_proj", help="comma list of modules to adapt")
    ap.add_argument("--num-layers", type=int, default=0, help="LoRA on the last N layers (0 = all)")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=12288, help="longer examples are skipped")
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--steps-per-eval", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="use at most N examples per split (0 = all)")
    ap.add_argument("--stop-token", default="<|tool_response>",
                    help="appended to each prompt / completion pair's completion; must be a single token")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        import mlx.optimizers as optim
        from mlx_lm import load
        from mlx_lm.tuner.datasets import CacheDataset
        from mlx_lm.tuner.trainer import TrainingArgs, train
        from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
    except ImportError as exc:
        raise SystemExit(f"g4kit-train-mlx needs mlx-lm on Apple Silicon (pip install '.[mlx]'): {exc}") from None

    t0 = time.time()
    model, tok = load(a.base)
    print(f"loaded {a.base} in {time.time() - t0:.0f}s", flush=True)
    stop_ids = tok.encode(a.stop_token, add_special_tokens=False)
    if len(stop_ids) != 1:
        raise SystemExit(f"--stop-token {a.stop_token!r} is not a single token: {stop_ids}")
    print("stop token", a.stop_token, stop_ids, flush=True)

    model.freeze()
    n_layers = a.num_layers or len(model.layers)
    lora_cfg = {"rank": a.rank, "scale": a.scale, "dropout": 0.0, "keys": a.keys.split(",")}
    linear_to_lora_layers(model, n_layers, lora_cfg)
    print_trainable_parameters(model)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "adapter_config.json").write_text(json.dumps(
        {"base": a.base, "num_layers": n_layers, "lora_parameters": lora_cfg, "fine_tune_type": "lora",
         "stop_token": a.stop_token, "iters": a.iters, "learning_rate": a.lr, "max_len": a.max_len}, indent=2))

    data = Path(a.data)
    with open(data / "train.jsonl", encoding="utf-8") as f:
        seg = "segments" in json.loads(f.readline())
    if seg:
        train_set = Segments(data / "train.jsonl", tok, a.max_len, a.limit)
        val_set = Segments(data / "valid.jsonl", tok, a.max_len, a.limit)
    else:
        train_set = PreRendered(data / "train.jsonl", tok, stop_ids, a.max_len, a.limit)
        val_set = PreRendered(data / "valid.jsonl", tok, stop_ids, a.max_len, a.limit)
    args = TrainingArgs(batch_size=1, iters=a.iters, val_batches=a.val_batches, steps_per_report=10,
                        steps_per_eval=a.steps_per_eval, steps_per_save=a.save_every,
                        adapter_file=str(out / "adapters.safetensors"), max_seq_length=a.max_len, grad_checkpoint=True)
    opt = optim.AdamW(learning_rate=a.lr, weight_decay=0.0)
    if seg:
        train(model=model, args=args, optimizer=opt, train_dataset=train_set, val_dataset=val_set,
              loss=seg_loss, iterate_batches=seg_iterate_batches)
    else:
        train(model=model, args=args, optimizer=opt, train_dataset=CacheDataset(train_set),
              val_dataset=CacheDataset(val_set))
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
