"""Distillation LoRA run of Gemma 4 31B on a Kaggle TPU v5e-8 (Tunix + Qwix). Library: train_lib.py; the kernel
that build_kernel.py writes puts both files next to each other.

  data      the dataset folder from prep_data.py: trajectories kept by g4kit-curate, rendered by g4kit-render-distill
            --mask-error-turns tool; C2_MODE think (default) or nothink. Primary windows <= 12,288 tokens
            ({mode}_train.jsonl), fallback windows <= 8,192 tokens ({mode}8k_train.jsonl).
  lora      rank SMOKE_RANK (32), alpha SMOKE_ALPHA (32) on the first configuration of C2_CONFIGS that fits in HBM.
            A first session with q, o, gate, up, down on all 60 layers ran out of HBM at compile time with both
            window lengths (HLO temporaries 19.62G at 12,288 and 19.83G at 8,192 vs 15.75G), so the default list is
              a  q,o,gate,up,down  layers 30-59  12k windows (buckets 8,192 / 12,288)
              b  q,o,gate,up,down  layers 30-59  8k windows (bucket 8,192)
              c  q,o,down          layers 30-59  12k
              d  q,o,down          layers 30-59  8k
              e  q,o               layers 30-59  12k (an earlier full run's targets, which trained at 12,288 with rank 16)
              f  q,o               layers 30-59  8k (last resort)
  memory    for each configuration in turn: apply the LoRA (a new configuration first frees the previous one's
            device state with Trainer.drop_lora; the loaded base model is reused), compile train for every bucket
            (largest first) and eval, and run one zero-gradient probe step (lr 0). An XLA out-of-memory error moves
            on to the next configuration; every attempt is logged with its HLO temporaries (metrics.json 'probe').
            The configuration that fits is trained and exported ('chosen_config'; adapter_config.json target_modules
            and layers_to_transform follow the factors actually trained). A failure that is not an out-of-memory
            error (applying the LoRA, loading a render, compiling) is recorded in 'errors' and also moves on.
            C2_CONFIGS replaces the list: 'label:targets@layers@windows; ...', e.g. 'e:q,o@30-59@12k'
  epochs    C2_EPOCHS (2) passes, each a fresh seeded permutation of the windows with targets; 2 windows per step,
            both from the same length bucket
  lr        peak C2_LR (2e-4); C2_WARMUP (30) linear warmup steps, constant through the first C2_SWEEP_STEPS steps,
            then cosine to C2_END_LR_FRAC x peak at the step horizon. The learning rate is an argument of the compiled
            step, so the schedule and the sweep need no recompilation.
  sweep     C2_SWEEP=auto|on|off. Candidates C2_LR first, then the other C2_SWEEP_LRS (1e-4, 4e-4); each trains the
            same first C2_SWEEP_STEPS (300) steps from the same initial LoRA and fresh Adam state, then the masked
            validation loss on up to C2_MAX_VAL (64) held-out windows decides; the best state continues. auto runs
            the other candidates only if, at the step time measured on the first, the sweep plus C2_SWEEP_MIN_FRAC
            of the remaining plan fit before the guard; otherwise the first candidate (2e-4) simply continues.
  horizon   after the first C2_SWEEP_STEPS steps and at every checkpoint, the steps that still fit before the guard
            are estimated from measured step and checkpoint times; the cosine ends at min(plan, fit) and later plan
            steps are dropped, so the schedule completes inside the session.
  ckpt      every FULL_EVERY (1,000) windows: validation loss, PEFT adapter in /kaggle/working/ckpt_NNNNN/, fp32 LoRA
            and Adam state in /kaggle/working/resume/ (overwritten); per-layer export check on the first checkpoint
            and on ckpt_final
  stop      end of the horizon, or 8.5 h after launcher start minus C2_RESERVE_S (480 s): final validation,
            /kaggle/working/ckpt_final/, metrics, clean exit
"""
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("SMOKE_OUT_DIR", "/kaggle/working/train_out")
import train_lib as T  # noqa: E402

GUARD_S = float(os.environ.get("FULL_GUARD_S", 8.5 * 3600))
RESERVE_S = float(os.environ.get("C2_RESERVE_S", 480))
STOP_AT = T.T0 + GUARD_S - RESERVE_S
CKPT_ROOT = Path(os.environ.get("FULL_CKPT_ROOT", "/kaggle/working"))
EVERY = int(os.environ.get("FULL_EVERY", 1000))
MODE = os.environ.get("C2_MODE", "think")
PEAK_LR = float(os.environ.get("C2_LR", 2e-4))
WARMUP = int(os.environ.get("C2_WARMUP", 30))
END_LR_FRAC = float(os.environ.get("C2_END_LR_FRAC", 0.1))
EPOCHS = int(os.environ.get("C2_EPOCHS", 2))
SEED = int(os.environ.get("C2_SEED", 0))
SWEEP = os.environ.get("C2_SWEEP", "auto")
SWEEP_LRS = [float(x) for x in os.environ.get("C2_SWEEP_LRS", "2e-4,1e-4,4e-4").split(",")]
SWEEP_STEPS = int(os.environ.get("C2_SWEEP_STEPS", 300))
SWEEP_MIN_FRAC = float(os.environ.get("C2_SWEEP_MIN_FRAC", 1.0))
MAX_VAL = int(os.environ.get("C2_MAX_VAL", 64))
BUCKETS = {"12k": [int(x) for x in os.environ.get("C2_BUCKETS_12K", "8192,12288").split(",")],
           "8k": [int(x) for x in os.environ.get("C2_BUCKETS_8K", "8192").split(",")]}
DEFAULT_CONFIGS = ("a:q,o,gate,up,down@30-59@12k; b:q,o,gate,up,down@30-59@8k; c:q,o,down@30-59@12k; "
                   "d:q,o,down@30-59@8k; e:q,o@30-59@12k; f:q,o@30-59@8k")
CKPT_EST_S = float(os.environ.get("C2_CKPT_EST_S", 150))   # first guess of one checkpoint's cost until one is measured
MAX_WINDOWS = int(os.environ.get("FULL_MAX_WINDOWS", 0))     # 0 = all; local tests use a few
LOG_EVERY = int(os.environ.get("FULL_LOG_EVERY", 10))
log = T.log
M = T.METRICS


def parse_configs(spec):
    """'label:targets@layers@windows; ...' -> configuration dicts in the order they are tried. targets: keys of
    train_lib.TARGET_MODULES (q, o, gate, up, down); layers: 'lo-hi'; windows: 12k (the C2_MODE render, buckets
    C2_BUCKETS_12K) or 8k (the C2_MODE + '8k' render, buckets C2_BUCKETS_8K)."""
    out = []
    for item in (x.strip() for x in spec.split(";")):
        if not item:
            continue
        label, sep, rest = item.partition(":")
        targets, layers, windows = (x.strip() for x in rest.split("@"))
        targets = [t.strip() for t in targets.split(",") if t.strip()]
        T.lora_regex(layers, targets)                   # raises on unsupported targets
        lo, hi = (int(x) for x in layers.split("-"))
        if not (sep and label.strip() and windows in BUCKETS and 0 <= lo <= hi):
            raise ValueError(f"bad LoRA configuration {item!r}")
        out.append({"label": label.strip(), "targets": targets, "layers": layers, "windows": windows,
                    "variant": MODE if windows == "12k" else f"{MODE}8k", "buckets": BUCKETS[windows],
                    "hf_targets": sorted(T.TARGET_MODULES[t][3] for t in targets),
                    "layer_list": list(range(lo, hi + 1))})
    if not out or len({c["label"] for c in out}) != len(out):
        raise ValueError(f"bad configuration list {spec!r}")
    return out


def describe(c):
    return f"{c['label']} ({','.join(c['targets'])} on layers {c['layers']}, {c['variant']} buckets {c['buckets']})"


CONFIGS = parse_configs(os.environ.get("C2_CONFIGS", DEFAULT_CONFIGS))
CANDIDATES = [PEAK_LR] + [x for x in SWEEP_LRS if abs(x - PEAK_LR) > 1e-12]
M["c2_config"] = {"guard_s": GUARD_S, "reserve_s": RESERVE_S, "every_windows": EVERY, "mode": MODE,
                  "peak_lr": PEAK_LR, "warmup_steps": WARMUP, "end_lr_frac": END_LR_FRAC, "epochs": EPOCHS,
                  "seed": SEED, "sweep": SWEEP, "sweep_candidates": CANDIDATES, "sweep_steps": SWEEP_STEPS,
                  "sweep_min_frac": SWEEP_MIN_FRAC, "max_val": MAX_VAL, "buckets": BUCKETS, "batch": T.BATCH,
                  "rank": T.RANK, "alpha": T.ALPHA, "configs": [describe(c) for c in CONFIGS],
                  "max_windows": MAX_WINDOWS}
M["checkpoints"] = []
M["steps"] = {"loss": [], "step_s": [], "bucket": [], "targets": [], "grad_norm": [], "lr": []}
M["horizon_updates"] = []


def tokenize_lines(lines, tok, chunk=400):
    """Same ids as train_lib.tokenize_windows (encode each segment with add_special_tokens=False and concatenate),
    batched through the Rust tokenizer's parallel encode_batch."""
    out = []
    for c0 in range(0, len(lines), chunk):
        flags, texts = [], []
        for line in lines[c0:c0 + chunk]:
            segs = json.loads(line)["segments"]
            flags.append([int(b) for _, b in segs])
            texts.extend(t for t, _ in segs)
        enc = tok.encode_batch(texts, add_special_tokens=False)
        k = 0
        for fl in flags:
            toks, mask = [], []
            for f in fl:
                ids = enc[k].ids
                k += 1
                toks += ids
                mask += [f] * len(ids)
            out.append((np.asarray(toks, dtype="<i4"), np.asarray(mask, dtype=np.uint8)))
    return out


def bucketize(train_w, order, buckets):
    """Seeded order -> steps of BATCH windows from the same length bucket (smallest bucket that holds the window);
    a leftover window per bucket is paired with a dummy."""
    queues = {L: [] for L in buckets}
    steps = []
    for i in order:
        n = len(train_w[i][0])
        L = next((b for b in buckets if n <= b), buckets[-1])
        queues[L].append(int(i))
        if len(queues[L]) == T.BATCH:
            steps.append((L, queues[L]))
            queues[L] = []
    for L, q in queues.items():
        if q:
            steps.append((L, q + [None] * (T.BATCH - len(q))))
    return steps


def load_variant(data_dir, man, variant, tok):
    """Tokenize one rendered variant; check token/mask hashes against the manifest and the validation-row rule."""
    t = time.time()
    v = man["variants"][variant]
    tlines = open(data_dir / v["train"]).readlines()
    vlines = open(data_dir / v["valid"]).readlines()
    idx = list(range(len(vlines)))
    random.Random(0).shuffle(idx)
    rows = v["valid_rows"][:MAX_VAL]
    rule_ok = idx[:len(v["valid_rows"])] == v["valid_rows"]
    if MAX_WINDOWS:
        tlines = tlines[:MAX_WINDOWS]
    train_w = tokenize_lines(tlines, tok)
    val_w = tokenize_lines([vlines[i] for i in rows], tok)
    mism = []
    for name, ws, ref in (("train", train_w, v.get("check_train_first64", [])),
                          ("valid", val_w, v.get("check_valid", []))):
        for i, r in enumerate(ref[:len(ws)]):
            tk, mk = ws[i]
            if T.sha256(tk.tobytes()) != r["tokens_sha256"] or T.sha256(mk.tobytes()) != r["mask_sha256"]:
                mism.append(f"{name}{i}")
    info = {"variant": variant, "s": time.time() - t, "train_windows": len(train_w), "valid_windows": len(val_w),
            "valid_rows_match_random0_rule": rule_ok, "hash_mismatches": mism,
            "hash_checked": min(len(v.get("check_train_first64", [])), len(train_w)) +
            min(len(v.get("check_valid", [])), len(val_w)),
            "train_tokens": int(sum(len(w[0]) for w in train_w)),
            "train_targets": int(sum(int(w[1][1:].sum()) for w in train_w)),
            "valid_targets": int(sum(int(w[1][1:].sum()) for w in val_w)),
            "max_len": int(max(len(w[0]) for w in train_w))}
    return train_w, val_w, info


def make_plan(train_w, buckets):
    keep = [i for i, (_, m) in enumerate(train_w) if m[1:buckets[-1]].sum() > 0]
    plan = []
    for e in range(EPOCHS):
        order = [keep[j] for j in np.random.default_rng(SEED + e).permutation(len(keep))]
        plan += [(L, idxs, e) for L, idxs in bucketize(train_w, order, buckets)]
    return plan, len(train_w) - len(keep)


def probe(tr, buckets):
    """Compile train for every bucket (largest first), eval at the largest, and run one zero-target step with lr 0
    at the largest bucket. Returns None if everything fits, the exception on an XLA out-of-memory error."""
    import jax
    try:
        for L in sorted(buckets, reverse=True):
            tr.compiled("train", L)
        tr.compiled("eval", max(buckets))
        L = max(buckets)
        tok, tgt, tm = T.make_batch([(np.zeros(2, "<i4"), np.zeros(2, np.uint8))] * T.BATCH, L)
        c, _, _ = tr.compiled("train", L)
        if T.fake_oom(tr.config_label, "step"):
            for x in jax.tree.leaves(tr.lora) + jax.tree.leaves(tr.opt):   # a failed call consumes its donated state
                x.delete()
            raise T.fake_oom_error(tr.config_label, "step")
        with tr.mesh:
            tr.lora, tr.opt, loss, _, _ = c(tr.lora, tr.opt, tr.rest, *tr.put([tok, tgt, tm]), tr.lr_arg(0.0))
        float(loss)
        return None
    except Exception as e:
        if T.is_oom(e):
            return e
        raise


def lr_at(step, peak, hold_until, horizon):
    """1-based step: linear warmup, constant until hold_until, cosine to END_LR_FRAC x peak at horizon."""
    if step <= WARMUP:
        return peak * step / max(WARMUP, 1)
    if step <= hold_until or horizon is None:
        return peak
    prog = min(1.0, (step - hold_until) / max(horizon - hold_until, 1))
    return peak * (END_LR_FRAC + (1 - END_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * prog)))


def save_resume(tr, info):
    import jax
    d = CKPT_ROOT / "resume"
    d.mkdir(parents=True, exist_ok=True)
    arrays, paths = {}, []
    for prefix, tree in (("lora", tr.lora), ("opt", tr.opt)):
        for p, x in jax.tree_util.tree_flatten_with_path(tree)[0]:
            key = f"{prefix}_{len(paths):05d}"
            arrays[key] = np.asarray(jax.device_get(x))
            paths.append([key, jax.tree_util.keystr(p), str(arrays[key].dtype), list(arrays[key].shape)])
    np.savez(d / "state.npz", **arrays)
    (d / "state.json").write_text(json.dumps({**info, "leaves": paths}, indent=0))


def main():
    import jax
    from tokenizers import Tokenizer
    T.OUT.mkdir(parents=True, exist_ok=True)
    if T.INTERPRET:                      # CPU tests only: Pallas splash kernel in interpret mode
        T.patch_splash_interpret()
    log(f"c2 run start; stop at {(STOP_AT - T.T0) / 3600:.2f} h after launcher start; candidates {CANDIDATES}; "
        f"LoRA configurations in order: {'; '.join(describe(c) for c in CONFIGS)}")
    M["jax"] = {"version": jax.__version__, "count": jax.device_count(), "device_kind": jax.devices()[0].device_kind}
    import importlib.metadata as md
    M["versions"] = {p: md.version(p) for p in ("jax", "jaxlib", "libtpu", "flax", "qwix", "google-tunix", "optax",
                                                "tokenizers", "safetensors", "numpy") if T._has_dist(md, p)}
    M["hbm_limit_gb"] = T.mem_report("start")[0]["limit_gb"]
    want = int(os.environ.get("SMOKE_EXPECT_DEVICES", 0))
    if want and jax.device_count() != want:
        raise SystemExit(f"expected {want} devices, got {jax.device_count()}")
    T.save_metrics()

    # data (primary variant) ---------------------------------------------------------------------------------
    data_dir, model_dir = Path(T.DATA_DIR), Path(T.MODEL_DIR)
    man = json.load(open(data_dir / "manifest.json"))
    tok_path = model_dir / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))
    M["tokenizer"] = {"sha256": T.sha256(tok_path.read_bytes()), "manifest": man.get("tokenizer_json_sha256")}
    data = {}                            # tokenized renders by variant, kept so the fallback never re-tokenizes
    variant = CONFIGS[0]["variant"]
    data[variant] = load_variant(data_dir, man, variant, tok)
    M["data"] = data[variant][2]
    log("data", M["data"])
    T.save_metrics()

    # model -------------------------------------------------------------------------------------------------------
    cfg = T.tunix_config_from_hf(model_dir)
    mesh = jax.make_mesh(T.MESH_SHAPE, ("fsdp", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 2)
    tr = T.Trainer(cfg, mesh, model_dir)
    t = time.time()
    tr.load()
    M["load_s"] = time.time() - t
    log(f"load done in {M['load_s']:.0f}s; live: {T.live_stats()}")

    # LoRA configuration fallback: the first configuration whose compile and probe step fit is trained ------------
    M["probe"], M["lora_applied"] = [], []
    applied, init_state, chosen, linfo = None, None, None, None
    for c in CONFIGS:
        if time.time() > STOP_AT:
            raise SystemExit("guard reached while probing LoRA configurations")
        key = (tuple(c["targets"]), c["layers"])
        t = time.time()
        err, phase = None, "lora"
        try:
            if applied != key:
                if applied is not None or getattr(tr, "model", None) is not None:
                    init_state = None
                    n = tr.drop_lora()
                    T.mem_report(f"after dropping LoRA {applied}")
                    log(f"dropped LoRA {applied}: {n} device arrays deleted; live: {T.live_stats()}")
                applied = None
                linfo = tr.apply_lora(layers=c["layers"], targets=c["targets"])
                M["lora"] = {"config": c["label"], "s": time.time() - t,
                             **{k: v for k, v in linfo.items() if k != "layers"},
                             "layers": [linfo["layers"][0], linfo["layers"][-1], len(linfo["layers"])],
                             "live_gb": T.live_gb()}
                M["lora_applied"].append(M["lora"])
                log("lora", M["lora"])
                tr.build()
                init_state = tr.host_state()  # initial LoRA (B = 0) + fresh Adam state: every sweep candidate starts here
                applied = key
            tr.config_label = c["label"]
            phase = "data"
            if c["variant"] not in data:
                data[c["variant"]] = load_variant(data_dir, man, c["variant"], tok)
                log(f"data ({c['variant']})", data[c["variant"]][2])
            phase = "probe"
            t = time.time()
            err = probe(tr, c["buckets"])    # None, or the XLA out-of-memory error
        except Exception as e:               # any other failure also moves on to the next configuration
            err = e
            T.record_error(f"config {c['label']} {phase}", e)
        rec = {"config": c["label"], "targets": c["hf_targets"], "layers": c["layers"], "variant": c["variant"],
               "buckets": c["buckets"], "s": time.time() - t, "fits": err is None,
               "phase": None if err is None else phase,
               "error_kind": None if err is None else ("oom" if T.is_oom(err) else "error"),
               "oom": None if err is None else str(err)[:2000]}
        if err is None:
            ma = T.mem_analysis(tr._compiled[("train", max(c["buckets"]))])
            rec["temporaries_gb"] = ma.get("temp_size_in_bytes")
            rec["arguments_gb"] = ma.get("argument_size_in_bytes")
        else:
            rec["temporaries_gb"] = T.oom_gb(err)
        M["probe"].append(rec)
        T.mem_report(f"after probe {c['label']}")
        if applied == key:
            tr.load_state(init_state)    # the probe step may have consumed the donated state; also resets Adam
        T.save_metrics()
        tgb = rec["temporaries_gb"]
        verdict = "FITS" if err is None else "OUT OF MEMORY" if rec["error_kind"] == "oom" else f"ERROR in {phase}"
        log(f"ATTEMPT {describe(c)}: {verdict}; HLO temporaries "
            f"{'%.2fG' % tgb if isinstance(tgb, float) else 'not reported'}"
            f"{'' if err is None else ' (' + str(err)[:300] + ')'}; {rec['s']:.0f}s")
        if err is None:
            chosen = c
            break
    if chosen is None:
        raise SystemExit("out of memory in every LoRA configuration: " + ", ".join(
            f"{r['config']} {r['temporaries_gb']}" for r in M["probe"]))
    variant, buckets = chosen["variant"], chosen["buckets"]
    train_w, val_w, info = data[variant]
    data.clear()
    M["data"] = info
    CH = {k: chosen[k] for k in ("label", "targets", "hf_targets", "layers", "windows", "variant", "buckets")}
    CH.update({"layer_list": linfo["layers"], "rank": T.RANK, "alpha": T.ALPHA,
               "temporaries_gb": M["probe"][-1]["temporaries_gb"], "attempts": len(M["probe"])})
    M["chosen_config"] = CH
    log(f"CHOSEN LoRA configuration {describe(chosen)}: PEFT target_modules {CH['hf_targets']}, layers_to_transform "
        f"{CH['layer_list'][0]}-{CH['layer_list'][-1]} ({len(CH['layer_list'])} layers), rank {T.RANK}, alpha {T.ALPHA}")
    T.save_metrics()
    M["compile"] = {f"{k}@{L}": T.mem_analysis(v) for (k, L), v in tr._compiled.items()
                    if not isinstance(v, Exception)}
    plan, no_target = make_plan(train_w, buckets)
    M["plan"] = {"variant": variant, "buckets": buckets, "steps": len(plan), "no_target_dropped": no_target,
                 "steps_per_bucket": {str(L): sum(1 for x in plan if x[0] == L) for L in buckets},
                 "steps_per_epoch": [sum(1 for x in plan if x[2] == e) for e in range(EPOCHS)]}
    log("plan", M["plan"])
    T.save_metrics()

    S = M["steps"]
    state = {"step": 0, "windows": 0, "horizon": None, "peak": PEAK_LR, "ckpt_s": [], "first_verify_done": False}

    def checkpoint(name, verify):
        t0 = time.time()
        v = tr.evaluate(val_w, buckets[-1])
        rec = {"name": name, "step": state["step"], "windows": state["windows"], "val_loss": v["loss"],
               "val_targets": sum(v["counts"]), "val_s": time.time() - t0, "elapsed_h": (time.time() - T.T0) / 3600,
               "lr": S["lr"][-1] if S["lr"] else None}
        if state["step"] > 0:
            tensors = T.lora_numpy(tr.lora)
            # adapter_config.json describes the factors in the file: layers_to_transform and target_modules come from
            # the trained tensors and must equal the chosen configuration (a difference is recorded as an error)
            layers = sorted({k[0] for k in tensors})
            targets = sorted({T.TUNIX_TO_HF[k[1]][1] for k in tensors})
            if layers != CH["layer_list"] or targets != CH["hf_targets"]:
                T.record_error("export", ValueError(f"{name}: adapter holds layers {layers} targets {targets}; chosen "
                                                    f"configuration {CH['label']} = {CH['layer_list']} {CH['hf_targets']}"))
            d = T.save_peft(T.to_peft(tensors, T.RANK), CKPT_ROOT / name, T.RANK, T.ALPHA, layers,
                            target_modules=targets)
            rec["dir"] = str(d)
            rec["bytes"] = (d / "adapter_model.safetensors").stat().st_size
            rec["tensors"] = len(tensors)
            rec["config"] = CH["label"]
            if verify:
                saved, _ = T.read_safetensors_f32(d / "adapter_model.safetensors")
                vl = [L for L in T.VERIFY_LAYERS if L in layers] or [layers[0], layers[-1]]
                checks = {L: T.verify_layer(tr, tensors, saved, L, T.RANK, T.ALPHA) for L in vl}
                worst = max(r["delta_rel_err"] for c in checks.values() for r in c.values())
                rec["verify"] = {"worst_delta_rel_err": worst, "pass": bool(worst < 2e-2 and all(
                    r.get("hf_weight_equal", True) for c in checks.values() for r in c.values())), "checks": checks}
            del tensors
            try:
                rec["skipped_nonfinite_updates"] = int(jax.device_get(tr.opt.total_notfinite))
            except Exception:
                rec["skipped_nonfinite_updates"] = None
            save_resume(tr, {"step": state["step"], "windows": state["windows"], "seed": SEED, "variant": variant,
                             "peak_lr": state["peak"], "horizon": state["horizon"],
                             "config": {k: CH[k] for k in ("label", "targets", "layers", "rank", "alpha")},
                             "order_note": "per epoch e: numpy default_rng(seed + e).permutation over windows with "
                                           "targets, then bucketize(); the step index resumes the plan"})
        rec["s"] = time.time() - t0
        if state["step"] > 0:
            state["ckpt_s"].append(rec["s"])
        M["checkpoints"].append(rec)
        T.save_metrics()
        log(f"CKPT {name}: step {state['step']} windows {state['windows']} val {v['loss']:.4f} "
            f"({rec['s']:.0f}s){' verify ' + str(rec['verify']['pass']) if 'verify' in rec else ''}")
        return v["loss"]

    def run_steps(first, last, peak, hold, tag, ckpt=True):
        """Train plan steps first..last (1-based) at the schedule for `peak`; returns step times; stops at the guard
        or at the current horizon. Checkpoints every EVERY windows when ckpt is True."""
        times = []
        step = first
        while step <= last:
            if state["horizon"] is not None and step > state["horizon"]:
                break
            if time.time() > STOP_AT:
                M["stopped_by_guard"] = True
                log(f"GUARD: stopping before step {step}")
                break
            L, idxs, _ = plan[step - 1]
            group = [train_w[i] if i is not None else (np.zeros(2, "<i4"), np.zeros(2, np.uint8)) for i in idxs]
            tok_b, tgt, tm = T.make_batch(group, L)
            lr = lr_at(step, peak, hold, state["horizon"])
            c, _, _ = tr.compiled("train", L)
            ts = time.time()
            with mesh:
                tr.lora, tr.opt, loss, _, gn = c(tr.lora, tr.opt, tr.rest, *tr.put([tok_b, tgt, tm]), tr.lr_arg(lr))
            loss = float(loss)
            dt = time.time() - ts
            times.append(dt)
            state["step"] = step
            state["windows"] += sum(i is not None for i in idxs)
            S["loss"].append(round(loss, 5))
            S["step_s"].append(round(dt, 3))
            S["bucket"].append(L)
            S["targets"].append(int(tm.sum()))
            S["grad_norm"].append(round(float(gn), 5))
            S["lr"].append(float(lr))
            if step % LOG_EVERY == 0:
                k = min(LOG_EVERY, len(times))
                log(f"{tag} step {step}/{state['horizon'] or len(plan)} win {state['windows']} loss "
                    f"{np.mean(S['loss'][-k:]):.4f} lr {lr:.2e} gnorm {np.mean(S['grad_norm'][-k:]):.3f} "
                    f"{np.mean(times[-k:]):.2f}s/step elapsed {(time.time() - T.T0) / 3600:.2f}h")
            if step % 50 == 0:
                T.save_metrics()
            if ckpt and state["windows"] >= state["next_ckpt"] and step < (state["horizon"] or len(plan)):
                checkpoint(f"ckpt_{state['windows']:05d}", verify=not state["first_verify_done"])
                state["first_verify_done"] = True
                state["next_ckpt"] += EVERY
                update_horizon(f"after ckpt at step {step}")
            step += 1
        return times

    def update_horizon(why):
        """Steps that still fit before STOP_AT at the measured costs; the cosine ends there (never past the plan)."""
        recent = S["step_s"][-300:]
        if len(recent) < 2:
            return
        t_step = float(np.mean(recent))
        ck = float(np.mean(state["ckpt_s"])) if state["ckpt_s"] else CKPT_EST_S
        left = STOP_AT - time.time()
        per_step = t_step + T.BATCH * ck / EVERY
        fit = int(0.97 * left / per_step)
        new = min(len(plan), state["step"] + max(fit, 0))
        old = state["horizon"]
        state["horizon"] = max(new, state["step"])
        M["horizon_updates"].append({"why": why, "step": state["step"], "t_step": t_step, "ckpt_s": ck,
                                     "left_s": left, "fit_steps": fit, "old": old, "new": state["horizon"],
                                     "plan": len(plan)})
        log(f"horizon {old} -> {state['horizon']} of {len(plan)} ({why}; {t_step:.2f}s/step, ckpt {ck:.0f}s, "
            f"{left / 3600:.2f}h left)")

    # base validation ---------------------------------------------------------------------------------------------
    checkpoint("ckpt_00000_base", verify=False)
    state["next_ckpt"] = EVERY
    n_sw = min(SWEEP_STEPS, len(plan))

    # first candidate (= the single-LR run if the sweep does not run) ------------------------------------------------
    sweep = {"mode": SWEEP, "steps": n_sw, "candidates": []}
    M["sweep"] = sweep
    t = time.time()
    times = run_steps(1, n_sw, CANDIDATES[0], n_sw, f"lr{CANDIDATES[0]:.0e}", ckpt=False)
    done_first = state["step"]
    v1 = tr.evaluate(val_w, buckets[-1])["loss"]
    sweep["candidates"].append({"lr": CANDIDATES[0], "val_loss": v1, "steps": done_first, "s": time.time() - t,
                                "median_step_s": float(np.median(times[2:] or times)) if times else None})
    log(f"SWEEP candidate lr {CANDIDATES[0]:.1e}: val {v1:.4f} after {done_first} steps")
    t_step = float(np.mean(times[2:] or times)) if times else 0.0
    left = STOP_AT - time.time()
    rest_steps = len(plan) - done_first
    val_s = sweep["candidates"][0]["s"] - sum(times)
    sweep_cost = (len(CANDIDATES) - 1) * (n_sw * t_step + max(val_s, 0) + 30)
    need = SWEEP_MIN_FRAC * rest_steps * (t_step + T.BATCH * CKPT_EST_S / EVERY)
    run_sweep = len(CANDIDATES) > 1 and done_first == n_sw and (
        SWEEP == "on" or (SWEEP == "auto" and sweep_cost + need <= left))
    sweep.update({"t_step": t_step, "left_s": left, "sweep_cost_s": sweep_cost, "need_s": need, "ran": run_sweep})
    log(f"SWEEP decision: {'run' if run_sweep else 'skip'} (sweep {sweep_cost / 3600:.2f}h + remaining plan "
        f"{need / 3600:.2f}h vs {left / 3600:.2f}h left, mode {SWEEP})")
    if run_sweep:
        states = {CANDIDATES[0]: (v1, tr.host_state(), {k: list(S[k]) for k in S})}
        for lr0 in CANDIDATES[1:]:
            tr.load_state(init_state)
            for k in S:
                S[k].clear()
            state.update({"step": 0, "windows": 0})
            t = time.time()
            times = run_steps(1, n_sw, lr0, n_sw, f"lr{lr0:.0e}", ckpt=False)
            v = tr.evaluate(val_w, buckets[-1])["loss"]
            sweep["candidates"].append({"lr": lr0, "val_loss": v, "steps": state["step"], "s": time.time() - t,
                                        "median_step_s": float(np.median(times[2:] or times)) if times else None})
            log(f"SWEEP candidate lr {lr0:.1e}: val {v:.4f} after {state['step']} steps")
            states[lr0] = (v, tr.host_state(), {k: list(S[k]) for k in S})
        best = min(states, key=lambda k: (states[k][0], abs(math.log(k / PEAK_LR))))
        sweep["best_lr"] = best
        sweep["train_loss_curves"] = {f"{k:.1e}": states[k][2]["loss"] for k in states}
        tr.load_state(states[best][1])
        for k in S:
            S[k][:] = states[best][2][k]
        state.update({"step": n_sw, "windows": sum(sum(i is not None for i in plan[s][1]) for s in range(n_sw)),
                      "peak": best})
        del states
        log(f"SWEEP best lr {best:.1e}")
    else:
        sweep["best_lr"] = CANDIDATES[0]
    T.save_metrics()

    # main run ------------------------------------------------------------------------------------------------------
    while state["next_ckpt"] <= state["windows"]:
        state["next_ckpt"] += EVERY
    update_horizon("after sweep window")
    t_loop = time.time()
    run_steps(state["step"] + 1, len(plan), state["peak"], n_sw, "train")
    M["stopped_by_guard"] = M.get("stopped_by_guard", False)
    M["train_s"] = time.time() - t_loop
    checkpoint("ckpt_final", verify=True)
    st = np.array(S["step_s"])
    b = np.array(S["bucket"])
    M["summary"] = {
        "config": {k: CH[k] for k in ("label", "hf_targets", "layers", "variant", "buckets", "rank", "alpha")},
        "attempts": [(r["config"], "fits" if r["fits"] else r["error_kind"], r["temporaries_gb"]) for r in M["probe"]],
        "variant": variant, "buckets": buckets, "steps_done": state["step"], "steps_planned": len(plan),
        "horizon": state["horizon"], "windows_done": state["windows"],
        "epochs_done": round(state["windows"] / max(1, M["data"]["train_windows"] - M["plan"]["no_target_dropped"]), 3),
        "stopped_by_guard": M["stopped_by_guard"], "train_h": M["train_s"] / 3600, "peak_lr": state["peak"],
        "sweep_ran": sweep["ran"], "sweep": [(c["lr"], round(c["val_loss"], 4)) for c in sweep["candidates"]],
        "median_step_s": {str(L): float(np.median(st[b == L][1:])) for L in buckets if (b == L).sum() > 1},
        "val": [(c["name"], c["windows"], round(c["val_loss"], 4)) for c in M["checkpoints"]],
        "final_verify_pass": M["checkpoints"][-1].get("verify", {}).get("pass"),
        "errors": [e["phase"] for e in M["errors"]]}
    M["done"] = True
    T.save_metrics()
    log("SUMMARY " + json.dumps(M["summary"]))
    log("c2 run done")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        T.record_error("main", e)
        sys.exit(1)
