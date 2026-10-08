"""Gemma 4 31B LoRA on a Kaggle TPU v5e-8 with Tunix (model) and Qwix (LoRA): the library behind train.py.

  targets   q (attn/q_einsum), o (attn/attn_vec_einsum) and the MLP gate_proj / up_proj / down_proj (nnx.Linear) on
            every layer (SMOKE_LORA_LAYERS default 0-59, C2_TARGETS default q,o,gate,up,down; train.py passes the
            layers and targets of each configuration it tries). k and v are not supported: Tunix fuses them, and on
            the 10 global layers vLLM loads the K weights into both the K and V slots, so a k_proj adapter would
            train one function and serve another.
  rank      SMOKE_RANK default 32, SMOKE_ALPHA default 32 (scale alpha / r; LoRA Without Regret's convention, the one
            its learning-rate guidance assumes)
  optimizer Adam moments only (optax.scale_by_adam inside apply_if_finite); the learning rate is an argument of
            the compiled train step, so schedules and sweeps need no recompilation
  export    PEFT keys base_model.model.model.language_model.layers.{L}.self_attn.{q,o}_proj... and
            ...layers.{L}.mlp.{gate,up,down}_proj.lora_{A,B}.weight; nnx.Linear kernels are [in, out], so
            lora_A = a.T [r, in] and lora_B = b.T [out, r]
  memory    is_oom() recognizes XLA out-of-memory errors so the caller can fall back to another LoRA configuration or
            shorter windows; oom_gb() reads the required temporaries from the message. Trainer.drop_lora() frees a
            configuration's device state so apply_lora(layers=..., targets=...) can try the next one on the loaded
            base model. C2_FAKE_OOM (CPU tests of these paths), comma-separated: L = the train@L compile of every
            configuration fails like an OOM; X = every train compile of configuration X; X@L = its train@L compile;
            X@step = its probe step fails at run time (the donated state is deleted first, as a real failure would)

Runs inside the venv the launcher builds (build_kernel.py embeds this file in the kernel). It also runs on CPU with a
tiny HF-format checkpoint, driven by the same environment variables (SMOKE_INTERPRET=1 runs the Pallas splash
attention kernel in interpret mode).

The SMOKE_, FULL_ and C2_ prefixes of the settings name the runs that introduced them: a smoke test, a full
distillation run and a curated run (c2).
"""
import hashlib
import json
import os
import re
import time
import traceback
from pathlib import Path

import numpy as np

T0 = float(os.environ.get("SMOKE_T0") or time.time())
GUARD_S = float(os.environ.get("SMOKE_GUARD_S", 90 * 60))
RESERVE_S = float(os.environ.get("SMOKE_RESERVE_S", 6 * 60))   # kept for export + metrics after the last phase
DEADLINE = T0 + GUARD_S - RESERVE_S

MODEL_DIR = os.environ.get("SMOKE_MODEL_DIR", "")
DATA_DIR = os.environ.get("SMOKE_DATA_DIR", "")
OUT = Path(os.environ.get("SMOKE_OUT_DIR", "/kaggle/working/smoke_out"))
SEQ_LONG = int(os.environ.get("SMOKE_SEQ_LONG", 12288))
SEQ_SHORT = int(os.environ.get("SMOKE_SEQ_SHORT", 8192))
CHUNK = int(os.environ.get("SMOKE_CHUNK", 1024))
STEPS = [int(x) for x in os.environ.get("SMOKE_STEPS", "10,10,10").split(",")]
BATCH = int(os.environ.get("SMOKE_BATCH", 2))
MESH_SHAPE = tuple(int(x) for x in os.environ.get("SMOKE_MESH", "2,4").split(","))
FLASH = os.environ.get("SMOKE_FLASH", "1") == "1"
INTERPRET = os.environ.get("SMOKE_INTERPRET", "0") == "1"          # CPU tests: Pallas splash kernel in interpret mode
FLASH_BLOCK = int(os.environ.get("SMOKE_FLASH_BLOCK", 0))           # 0 = Tunix default (1024)
LR = float(os.environ.get("SMOKE_LR", 2e-4))
RANK = int(os.environ.get("SMOKE_RANK", 32))
ALPHA = float(os.environ.get("SMOKE_ALPHA", 32.0))
LORA_LAYERS = os.environ.get("SMOKE_LORA_LAYERS", "0-59")
TARGETS = [t.strip() for t in os.environ.get("C2_TARGETS", "q,o,gate,up,down").split(",") if t.strip()]
VERIFY_LAYERS = [int(x) for x in os.environ.get("SMOKE_VERIFY_LAYERS", "0,5,30,35,59").split(",")]
def parse_fake_oom(spec):
    """C2_FAKE_OOM -> ({bucket lengths}, [(configuration label, None | bucket length | 'step')])."""
    lengths, rules = set(), []
    for tok in (t.strip() for t in spec.split(",")):
        if tok.isdigit():
            lengths.add(int(tok))
        elif tok:
            lab, _, at = tok.partition("@")
            rules.append((lab, None if not at else at if at == "step" else int(at)))
    return lengths, rules


FAKE_OOM, FAKE_OOM_RULES = parse_fake_oom(os.environ.get("C2_FAKE_OOM", ""))
MAX_VAL = int(os.environ.get("SMOKE_MAX_VAL", 24))
PHASES = set(os.environ.get("SMOKE_PHASES", "tokenize,load,lora,val0,train,val1,export").split(","))
COMPUTE_DTYPE_NAME = os.environ.get("SMOKE_DTYPE", "bfloat16")
PEFT_BASE_NAME = "google/gemma-4-31b-it-qat-w4a16-ct"

METRICS = {"t0_unix": T0, "guard_s": GUARD_S, "phases": {}, "errors": [], "config": {
    "seq_long": SEQ_LONG, "seq_short": SEQ_SHORT, "chunk": CHUNK, "steps": STEPS, "batch": BATCH,
    "mesh": MESH_SHAPE, "flash": FLASH, "flash_block": FLASH_BLOCK, "lr": LR, "rank": RANK, "alpha": ALPHA,
    "lora_layers": LORA_LAYERS, "targets": TARGETS, "dtype": COMPUTE_DTYPE_NAME}}


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.2f}m]", *a, flush=True)


def time_left():
    return DEADLINE - time.time()


def save_metrics():
    OUT.mkdir(parents=True, exist_ok=True)
    METRICS["elapsed_s"] = time.time() - T0
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(METRICS, indent=1, default=str))
    tmp.replace(OUT / "metrics.json")


def record_error(phase, e):
    tb = traceback.format_exc()
    log(f"ERROR in {phase}: {type(e).__name__}: {e}\n{tb}")
    METRICS["errors"].append({"phase": phase, "type": type(e).__name__, "msg": str(e)[:4000], "tb": tb[-6000:]})
    save_metrics()


# ----------------------------------------------------------------------------------------------- data
def sha256(b):
    return hashlib.sha256(b).hexdigest()


def tokenize_windows(path, tok):
    out = []
    for line in open(path):
        toks, mask = [], []
        for text, train in json.loads(line)["segments"]:
            ids = tok.encode(text, add_special_tokens=False).ids
            toks += ids
            mask += [int(train)] * len(ids)
        out.append((np.asarray(toks, dtype="<i4"), np.asarray(mask, dtype=np.uint8)))
    return out


def make_batch(windows, L):
    """windows: list of (tokens, mask); right-pad with 0 to L (truncate longer). Position p predicts token p+1."""
    B = len(windows)
    tok = np.zeros((B, L), np.int32)
    tgt = np.zeros((B, L), np.int32)
    tm = np.zeros((B, L), np.float32)
    for i, (t, m) in enumerate(windows):
        t, m = t[:L], m[:L]
        n = len(t)
        tok[i, :n] = t
        tgt[i, :n - 1] = t[1:]
        tm[i, :n - 1] = m[1:]
    return tok, tgt, tm


# ----------------------------------------------------------------------------------------------- model
def tunix_config_from_hf(model_dir):
    """Tunix ModelConfig built from the HF config.json (text part), checked against gemma4_31b() when it applies."""
    from tunix.models.gemma4 import model as g4
    raw = json.load(open(Path(model_dir) / "config.json"))
    tc = raw.get("text_config", raw)
    types = tc["layer_types"]
    period = None
    for p in range(1, len(types) + 1):
        if all(types[i] == types[i % p] for i in range(len(types))):
            period = p
            break
    amap = {"sliding_attention": g4.AttentionType.LOCAL_SLIDING, "full_attention": g4.AttentionType.GLOBAL}
    rp = tc["rope_parameters"]
    cfg = g4.ModelConfig(
        num_layers=tc["num_hidden_layers"], num_embed=tc["vocab_size"], embed_dim=tc["hidden_size"],
        hidden_dim=tc["intermediate_size"], num_heads=tc["num_attention_heads"], head_dim=tc["head_dim"],
        num_kv_heads=tc["num_key_value_heads"], num_global_kv_heads=tc.get("num_global_key_value_heads"),
        global_key_size=tc.get("global_head_dim") or tc["head_dim"], sliding_window_size=tc["sliding_window"],
        k_eq_v_global=bool(tc.get("attention_k_eq_v", False)),
        final_logit_softcap=tc.get("final_logit_softcapping"),
        attention_pattern=tuple(amap[t] for t in types[:period]),
        global_rope_proportion=float(rp["full_attention"].get("partial_rotary_factor", 1.0)),
        local_rope_proportion=float(rp["sliding_attention"].get("partial_rotary_factor", 1.0)),
        local_base_frequency=int(rp["sliding_attention"]["rope_theta"]),
        global_base_frequency=int(rp["full_attention"]["rope_theta"]),
        per_layer_input_dim=int(tc.get("hidden_size_per_layer_input") or 0),
    )
    assert not tc.get("enable_moe_block"), "MoE not expected"
    assert int(tc.get("num_kv_shared_layers") or 0) == 0
    assert rp["full_attention"].get("rope_type", "default") in ("proportional", "default")
    if tc["num_hidden_layers"] == 60 and tc["hidden_size"] == 5376:
        ref = g4.ModelConfig.gemma4_31b()
        for k in ("num_layers", "num_embed", "embed_dim", "hidden_dim", "num_heads", "head_dim", "num_kv_heads",
                  "num_global_kv_heads", "global_key_size", "sliding_window_size", "k_eq_v_global",
                  "final_logit_softcap", "attention_pattern", "global_rope_proportion", "local_rope_proportion",
                  "local_base_frequency", "global_base_frequency"):
            assert getattr(cfg, k) == getattr(ref, k), (k, getattr(cfg, k), getattr(ref, k))
    return cfg


def set_mode(model, flash, remat, block=0):
    """Mutate every distinct ModelConfig object reachable from the model (they are shared by reference)."""
    from tunix.models.gemma4 import model as g4
    seen = {}
    for _, m in model.iter_modules():
        c = getattr(m, "config", None)
        if c is not None and hasattr(c, "use_flash_attention") and id(c) not in seen:
            seen[id(c)] = c
    for c in seen.values():
        c.use_flash_attention = flash
        c.remat_config = g4.RematConfig.DECODER if remat else g4.RematConfig.NONE
        if block:
            c.flash_attention_block_size = block
    return len(seen)


# target -> (Tunix module path under layers/N, Qwix weight name, HF/PEFT parent, HF/PEFT module name)
TARGET_MODULES = {"q": ("attn/q_einsum", "w", "self_attn", "q_proj"),
                  "o": ("attn/attn_vec_einsum", "w", "self_attn", "o_proj"),
                  "gate": ("mlp/gate_proj", "kernel", "mlp", "gate_proj"),
                  "up": ("mlp/up_proj", "kernel", "mlp", "up_proj"),
                  "down": ("mlp/down_proj", "kernel", "mlp", "down_proj")}
TUNIX_TO_HF = {v[0].split("/")[1]: (v[2], v[3]) for v in TARGET_MODULES.values()}


def lora_regex(spec, targets=None):
    """Qwix module_path regex (re.fullmatch against 'layers/N/attn/q_einsum' style paths)."""
    targets = TARGETS if targets is None else targets
    bad = [t for t in targets if t not in TARGET_MODULES]
    if bad:
        raise ValueError(f"unsupported LoRA targets {bad} (k and v are deliberately unsupported)")
    lo, hi = (int(x) for x in spec.split("-"))
    mods = "|".join(TARGET_MODULES[t][0] for t in targets)
    return r"layers/(" + "|".join(str(i) for i in range(lo, hi + 1)) + r")/(" + mods + r")"


def is_oom(e):
    m = str(e)
    return "RESOURCE_EXHAUSTED" in m or "Ran out of memory" in m or "out of memory" in m.lower()


def oom_gb(msg):
    """Memory figure of an XLA out-of-memory message in G (as XLA prints it): the HLO temporaries of a failed compile
    ('... HLO temporaries (19.62G) exceeds available HBM ...'), else a 'Used X of Y' total; None if neither is there."""
    m = (re.search(r"HLO temporaries \(([\d.]+)([KMGT])", str(msg))
         or re.search(r"[Uu]sed ([\d.]+)([KMGT]) of", str(msg)))
    if not m:
        return None
    return round(float(m.group(1)) * {"K": 2.0 ** -20, "M": 2.0 ** -10, "G": 1.0, "T": 1024.0}[m.group(2)], 3)


def fake_oom(label, at):
    """True if C2_FAKE_OOM makes configuration `label` fail at `at`: a train bucket length (compile) or 'step' (the
    probe step at run time)."""
    if isinstance(at, int) and at in FAKE_OOM:
        return True
    return any(lab == label and (a == at or (a is None and at != "step")) for lab, a in FAKE_OOM_RULES)


def fake_oom_error(label, what):
    if what == "step":
        return RuntimeError(f"RESOURCE_EXHAUSTED: Error allocating device buffer: attempting to allocate 3.21G. That "
                            f"was not possible. (simulated by C2_FAKE_OOM for configuration {label}, probe step)")
    return RuntimeError(f"RESOURCE_EXHAUSTED: Ran out of memory on HBM, the total memory required for HLO temporaries "
                        f"(99.50G) exceeds available HBM (15.75G). (simulated by C2_FAKE_OOM for configuration {label}, "
                        f"{what})")


def live_gb():
    """Bytes of every live jax.Array (logical size, all devices together), in GiB."""
    import jax
    return sum(x.nbytes for x in jax.live_arrays()) / 2**30


def live_stats():
    import jax
    arrs = jax.live_arrays()
    return f"{len(arrs)} arrays, {sum(x.nbytes for x in arrs) / 2**30:.3f} GiB"


def patch_qwix_lora_dtype(dtype):
    """Qwix reads the LoRA params at every call; we keep fp32 masters and compute the LoRA path in `dtype`."""
    from qwix._src.providers import lora as ql
    if getattr(ql, "_smoke_patched", False):
        return
    orig = ql._get_or_create_lora_params

    def wrapped(*args, **kw):
        a, b = orig(*args, **kw)
        return a.astype(dtype), b.astype(dtype)

    ql._get_or_create_lora_params = wrapped
    ql._smoke_patched = True


def patch_splash_interpret():
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
    if getattr(splash, "_smoke_patched", False):
        return
    orig = splash.make_splash_mha

    def wrapped(*a, **kw):
        kw["interpret"] = True
        return orig(*a, **kw)

    splash.make_splash_mha = wrapped
    splash._smoke_patched = True


def mem_report(tag=""):
    import jax
    rows = []
    for d in jax.local_devices():
        try:
            s = d.memory_stats() or {}
        except Exception:
            s = {}
        rows.append({"dev": d.id, "in_use_gb": s.get("bytes_in_use", 0) / 2**30,
                     "peak_gb": s.get("peak_bytes_in_use", 0) / 2**30, "limit_gb": s.get("bytes_limit", 0) / 2**30})
    if rows and rows[0]["limit_gb"]:
        log(f"HBM {tag}: in_use max {max(r['in_use_gb'] for r in rows):.2f} GB, peak max "
            f"{max(r['peak_gb'] for r in rows):.2f} GB, limit {rows[0]['limit_gb']:.2f} GB")
    return rows


def mem_analysis(compiled):
    try:
        m = compiled.memory_analysis()
        keys = ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes",
                "generated_code_size_in_bytes")
        d = {k: getattr(m, k, None) for k in keys}
        d = {k: (v / 2**30 if isinstance(v, (int, float)) else v) for k, v in d.items()}
        if all(isinstance(d[k], float) for k in keys[:4]):
            d["est_peak_gb"] = (d["argument_size_in_bytes"] + d["output_size_in_bytes"] - d["alias_size_in_bytes"]
                                + d["temp_size_in_bytes"])
        return d
    except Exception as e:
        return {"error": str(e)}


class Trainer:
    def __init__(self, cfg, mesh, model_dir):
        import jax.numpy as jnp
        self.cfg, self.mesh, self.model_dir = cfg, mesh, model_dir
        self.dtype = getattr(jnp, COMPUTE_DTYPE_NAME)
        self.config_label = None             # name of the LoRA configuration being probed (C2_FAKE_OOM rules)
        self._compiled = {}

    def load(self):
        import jax
        from tunix.models.gemma4 import params_safetensors as g4p
        self.cfg.dtype = self.dtype
        self.cfg.param_dtype = self.dtype
        self.cfg.use_flash_attention = False
        cpu = jax.devices("cpu")[0]
        # Host-side preprocessing (q/k/v reshapes and stacks) would otherwise run on TPU 0 and hold ~10 GB there.
        with jax.default_device(cpu):
            self.base = g4p.create_model_from_safe_tensors(str(self.model_dir), self.cfg, self.mesh, dtype=None,
                                                           text_only=True)
        return self.base

    def apply_lora(self, seed=0, layers=None, targets=None):
        """LoRA on `targets` (keys of TARGET_MODULES) of the layers `layers` ('lo-hi'); the defaults are
        SMOKE_LORA_LAYERS and C2_TARGETS. Qwix adapts a clone of self.base that shares its weight buffers, so after
        drop_lora() another configuration can be applied to the same loaded model. Raises ValueError if the adapted
        (layer, module) pairs are not exactly the requested ones (restricted to the model's layers)."""
        import jax
        import qwix
        from flax import nnx
        layers = LORA_LAYERS if layers is None else layers
        targets = list(TARGETS if targets is None else targets)
        set_mode(self.base, flash=False, remat=False)
        provider = qwix.LoraProvider(
            module_path=lora_regex(layers, targets), rank=RANK, alpha=ALPHA,
            # mlx_lm LoRALinear: lora_a ~ U(-1/sqrt(in), 1/sqrt(in)); variance_scaling(1/3, fan_in, uniform) is that
            lora_a_initializer=jax.nn.initializers.variance_scaling(1.0 / 3.0, "fan_in", "uniform"))
        dummy = self.base.get_model_input()
        with self.mesh:
            self.model = qwix.apply_lora_to_model(self.base, provider, **dummy, rngs=nnx.Rngs(seed))
        n_cfg = set_mode(self.model, flash=FLASH, remat=True, block=FLASH_BLOCK)
        self.graphdef, lora, self.rest = nnx.split(self.model, nnx.LoRAParam, ...)
        self.lora = jax.tree.map(lambda x: x.astype(np.float32), lora)
        patch_qwix_lora_dtype(self.dtype)
        n = sum(int(np.prod(x.shape)) for x in jax.tree.leaves(self.lora))
        keys = [parse_lora_key(jax.tree_util.keystr(p)) for p, _ in jax.tree_util.tree_flatten_with_path(self.lora)[0]]
        got = {(k[0], k[1]) for k in keys if k}
        lo, hi = (int(x) for x in layers.split("-"))
        want = {(L, TARGET_MODULES[t][0].split("/")[1]) for L in range(lo, min(hi, self.cfg.num_layers - 1) + 1)
                for t in targets}
        if None in keys or got != want:
            raise ValueError(f"LoRA tree does not match layers {layers} targets {targets}: missing "
                             f"{sorted(want - got)[:6]}, unexpected {sorted(got - want)[:6]}")
        self.lora_layers, self.lora_targets = layers, targets
        found = sorted({k[0] for k in keys})
        shapes = {}
        for path, x in jax.tree_util.tree_flatten_with_path(self.lora)[0]:
            k = parse_lora_key(jax.tree_util.keystr(path))
            if k[0] in (found[0], found[-1]):
                shapes["%d.%s.%s" % k] = list(x.shape)
        return {"lora_params": n, "lora_tensors": len(jax.tree.leaves(self.lora)), "layers": found,
                "targets": targets, "shapes": shapes, "configs_mutated": n_cfg}

    def drop_lora(self):
        """Free the current LoRA configuration's device state: the LoRA factors inside the adapted model, their fp32
        masters, the optimizer state, the compiled steps and JAX's caches. The base weights are shared with the adapted
        clone and are never deleted (guarded by identity). Returns the number of device arrays deleted."""
        import gc

        import jax
        from flax import nnx
        keep = {id(x) for x in jax.tree.leaves(nnx.state(self.base))}
        leaves = []
        if getattr(self, "model", None) is not None:
            leaves += jax.tree.leaves(nnx.state(self.model, nnx.LoRAParam))
        for name in ("lora", "opt"):
            if getattr(self, name, None) is not None:
                leaves += jax.tree.leaves(getattr(self, name))
        n = 0
        for x in leaves:
            if id(x) in keep or not hasattr(x, "delete"):
                continue
            try:
                if not x.is_deleted():
                    x.delete()
                    n += 1
            except Exception:
                pass
        del leaves
        for name in ("model", "graphdef", "rest", "lora", "opt", "lora_sh", "opt_sh", "opt_init_j", "train_j", "eval_j",
                     "tx"):
            if hasattr(self, name):
                setattr(self, name, None)
        self._compiled = {}
        self.config_label = None
        jax.clear_caches()
        gc.collect()
        return n

    def build(self, tx=None):
        """tx: optax transformation producing Adam-normalized updates (default: apply_if_finite(scale_by_adam), i.e.
        AdamW without weight decay); the compiled train step multiplies them by -lr, its last argument."""
        import jax
        import jax.numpy as jnp
        import optax
        from flax import nnx
        from jax.sharding import NamedSharding, PartitionSpec as P
        mesh, graphdef = self.mesh, self.graphdef
        softcap = self.cfg.final_logit_softcap
        need_mask = not FLASH

        def hidden(lora, rest, tokens):
            m = nnx.merge(graphdef, lora, rest)
            B, L = tokens.shape
            pos = jnp.broadcast_to(jnp.arange(L, dtype=jnp.int32)[None, :], (B, L))
            mask = jnp.broadcast_to(jnp.tril(jnp.ones((L, L), jnp.bool_))[None], (B, L, L)) if need_mask else None
            h, _ = m(tokens, pos, None, mask, skip_lm_head=True)
            emb = m.embedder.input_embedding[...]
            emb = jax.lax.with_sharding_constraint(emb, NamedSharding(mesh, P("tp", None)))
            return h, emb

        def nll_sums(h, emb, targets, tmask):
            B, L, D = h.shape
            n = L // CHUNK
            hs = h.reshape(B, n, CHUNK, D).transpose(1, 0, 2, 3)
            ts = targets.reshape(B, n, CHUNK).transpose(1, 0, 2)
            ms = tmask.reshape(B, n, CHUNK).transpose(1, 0, 2)

            @jax.checkpoint
            def one(hc, tc, mc, e):
                z = jnp.einsum("bcd,vd->bcv", hc, e, preferred_element_type=jnp.float32)
                if softcap:
                    z = jnp.tanh(z / softcap) * softcap
                lse = jax.nn.logsumexp(z, axis=-1)
                zt = jnp.sum(jnp.where(jax.lax.broadcasted_iota(jnp.int32, z.shape, 2) == tc[..., None], z, 0.0),
                             axis=-1)
                return jnp.sum((lse - zt) * mc, axis=-1)

            def body(c, xs):
                return c + one(*xs, emb), None

            tot, _ = jax.lax.scan(body, jnp.zeros((B,), jnp.float32), (hs, ts, ms))
            return tot

        if tx is None:
            tx = optax.apply_if_finite(optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8), max_consecutive_errors=20)
        self.tx = tx

        def loss_fn(lora, rest, tokens, targets, tmask):
            h, emb = hidden(lora, rest, tokens)
            per = nll_sums(h, emb, targets, tmask)
            return per.sum() / jnp.maximum(tmask.sum(), 1.0), per

        def train_step(lora, opt, rest, tokens, targets, tmask, lr):
            (loss, per), g = jax.value_and_grad(loss_fn, has_aux=True)(lora, rest, tokens, targets, tmask)
            upd, opt = tx.update(g, opt, lora)
            upd = jax.tree.map(lambda u: (-lr * u).astype(u.dtype), upd)
            lora = optax.apply_updates(lora, upd)
            return lora, opt, loss, per, optax.global_norm(g)

        def eval_step(lora, rest, tokens, targets, tmask):
            h, emb = hidden(lora, rest, tokens)
            return nll_sums(h, emb, targets, tmask)

        rep = NamedSharding(mesh, P())
        self.rep = rep
        lora_sh = jax.tree.map(lambda x: x.sharding, self.lora)
        self.lora_sh = lora_sh

        def is_state(n):
            return isinstance(n, nnx.State)

        # Adam moments follow the LoRA shardings, scalars (count) are replicated over the mesh
        opt_sh = jax.tree.map(lambda n: lora_sh if is_state(n) else rep, jax.eval_shape(tx.init, self.lora),
                              is_leaf=is_state)
        self.opt_sh = opt_sh
        self.opt_init_j = jax.jit(tx.init, out_shardings=opt_sh)
        with mesh:
            self.opt = self.opt_init_j(self.lora)
        self.data_sh = NamedSharding(mesh, P("fsdp", None)) if MESH_SHAPE[0] > 1 else rep
        self.train_j = jax.jit(train_step, donate_argnums=(0, 1), out_shardings=(lora_sh, opt_sh, rep, rep, rep))
        self.eval_j = jax.jit(eval_step, out_shardings=rep)
        self._compiled = {}

    def put(self, arrs):
        import jax
        return [jax.device_put(a, self.data_sh) for a in arrs]

    def lr_arg(self, lr):
        import jax
        return jax.device_put(np.asarray(lr, np.float32), self.rep)

    def reset_opt(self):
        """Fresh optimizer state (Adam moments zero, counters zero) for the current LoRA params."""
        with self.mesh:
            self.opt = self.opt_init_j(self.lora)

    def host_state(self):
        """(LoRA, optimizer state) copied to host numpy, for sweeps and restores."""
        import jax
        return jax.device_get(self.lora), jax.device_get(self.opt)

    def load_state(self, host):
        """Replace the device LoRA / optimizer state with a host copy (shardings of the original state); the old
        device buffers are deleted first so HBM never holds two copies."""
        import jax
        lora_h, opt_h = host
        for x in jax.tree.leaves(self.lora) + jax.tree.leaves(self.opt):
            try:
                x.delete()
            except Exception:
                pass
        self.lora = jax.tree.map(lambda h, sh: jax.device_put(np.asarray(h), sh), lora_h, self.lora_sh)
        self.opt = jax.tree.map(lambda h, sh: jax.device_put(np.asarray(h), sh), opt_h, self.opt_sh)

    def compiled(self, kind, L):
        key = (kind, L)
        if key in self._compiled:
            if isinstance(self._compiled[key], Exception):
                raise RuntimeError(f"{kind}@{L} failed to compile earlier: {self._compiled[key]}")
            return self._compiled[key], 0.0, None
        z = [np.zeros((BATCH, L), np.int32), np.zeros((BATCH, L), np.int32), np.zeros((BATCH, L), np.float32)]
        args = self.put(z)
        t = time.time()
        try:
            if kind == "train" and fake_oom(self.config_label, L):
                raise fake_oom_error(self.config_label, f"train@{L}")
            with self.mesh:
                if kind == "train":
                    c = self.train_j.lower(self.lora, self.opt, self.rest, *args, self.lr_arg(0.0)).compile()
                else:
                    c = self.eval_j.lower(self.lora, self.rest, *args).compile()
        except Exception as e:
            self._compiled[key] = e
            log(f"compile {kind}@{L} FAILED after {time.time() - t:.0f}s: {str(e)[:3000]}")
            raise
        dt = time.time() - t
        ma = mem_analysis(c)
        log(f"compiled {kind}@{L} in {dt:.0f}s; memory_analysis {json.dumps(ma)}")
        self._compiled[key] = c
        return c, dt, ma

    def evaluate(self, windows, L):
        c, cdt, ma = self.compiled("eval", L)
        sums, counts, times = [], [], []
        for i in range(0, len(windows), BATCH):
            group = windows[i:i + BATCH]
            pad = BATCH - len(group)
            group = group + [(np.zeros(2, "<i4"), np.zeros(2, np.uint8))] * pad
            tok, tgt, tm = make_batch(group, L)
            t = time.time()
            with self.mesh:
                per = c(self.lora, self.rest, *self.put([tok, tgt, tm]))
            per = np.asarray(per)
            times.append(time.time() - t)
            for j in range(BATCH - pad):
                sums.append(float(per[j]))
                counts.append(float(tm[j].sum()))
        return {"sums": sums, "counts": counts, "loss": sum(sums) / max(sum(counts), 1.0),
                "batch_s": times, "compile_s": cdt, "mem_analysis": ma}

    def train(self, batches, L, tag):
        c, cdt, ma = self.compiled("train", L)
        rows = []
        for i, group in enumerate(batches):
            if time_left() < 60:
                log(f"{tag}: guard reached before step {i + 1}")
                break
            tok, tgt, tm = make_batch(group, L)
            args = self.put([tok, tgt, tm])
            t = time.time()
            with self.mesh:
                self.lora, self.opt, loss, per, gn = c(self.lora, self.opt, self.rest, *args, self.lr_arg(LR))
            loss = float(loss)
            dt = time.time() - t
            rows.append({"step_s": dt, "loss": loss, "grad_norm": float(gn), "targets": float(tm.sum()),
                         "tokens": int(sum(min(len(w[0]), L) for w in group)), "padded_tok_per_s": BATCH * L / dt})
            log(f"{tag} step {i + 1}/{len(batches)}: loss {loss:.4f} gnorm {float(gn):.3e} {dt:.2f}s "
                f"targets {int(tm.sum())}")
        return {"compile_s": cdt, "mem_analysis": ma, "steps": rows, "mem": mem_report(f"after {tag}")}


# ----------------------------------------------------------------------------------------------- export
LORA_KEY = re.compile(r"layers\W+(\d+)\W+(?:attn\W+(q_einsum|attn_vec_einsum)\W+w|"
                      r"mlp\W+(gate_proj|up_proj|down_proj)\W+kernel)_lora_([ab])")


def parse_lora_key(s):
    """(layer, Tunix module name, 'a' | 'b') from a LoRA param path, e.g. layers/3/mlp/up_proj/kernel_lora_a."""
    m = LORA_KEY.search(s)
    return (int(m.group(1)), m.group(2) or m.group(3), m.group(4)) if m else None


def lora_numpy(lora_state):
    import jax
    out = {}
    for path, x in jax.tree_util.tree_flatten_with_path(lora_state)[0]:
        k = parse_lora_key(jax.tree_util.keystr(path))
        if k is None:
            raise ValueError(f"unexpected LoRA leaf {jax.tree_util.keystr(path)}")
        out[k] = np.asarray(jax.device_get(x), dtype=np.float32)
    return out


def peft_key(L, name, ab):
    """name: HF module name (q_proj, o_proj, gate_proj, up_proj, down_proj)."""
    parent = "mlp" if name in ("gate_proj", "up_proj", "down_proj") else "self_attn"
    return f"base_model.model.model.language_model.layers.{L}.{parent}.{name}.lora_{ab.upper()}.weight"


def to_peft(tensors, rank):
    """Qwix factors -> PEFT matrices.
    q_einsum 'BTD,NDH->BTNH': a (D, r) -> lora_A [r, D] = a.T ; b (r, N, H) -> lora_B [N*H, r] = b.reshape(r, -1).T
    attn_vec 'BTNH,NHD->BTD': a (N, H, r) -> lora_A [r, N*H] = a.reshape(-1, r).T ; b (r, D) -> lora_B [D, r] = b.T
    Head-major flattening n*H + h matches the Tunix loader's reshape of HF q_proj [N*H, D] / o_proj [D, N*H].
    mlp gate/up/down (nnx.Linear, kernel [in, out] = HF weight.T, Qwix dot_general LoRA x @ a @ b):
             a (in, r) -> lora_A [r, in] = a.T ; b (r, out) -> lora_B [out, r] = b.T"""
    out = {}
    for (L, mod, ab), x in tensors.items():
        if mod == "q_einsum":
            name, mat = "q_proj", (x.T if ab == "a" else x.reshape(rank, -1).T)
        elif mod == "attn_vec_einsum":
            name, mat = "o_proj", (x.reshape(-1, rank).T if ab == "a" else x.T)
        elif mod in ("gate_proj", "up_proj", "down_proj"):
            assert x.ndim == 2, (L, mod, ab, x.shape)
            name, mat = mod, x.T
        else:
            raise ValueError(f"no PEFT mapping for {mod}")
        out[peft_key(L, name, ab)] = np.ascontiguousarray(mat)
    return out


def save_peft(peft, out_dir, rank, alpha, layers, target_modules=None):
    import jax
    import jax.numpy as jnp
    from safetensors.flax import save_file
    out_dir.mkdir(parents=True, exist_ok=True)
    cpu = jax.devices("cpu")[0]
    save_file({k: jax.device_put(jnp.asarray(v, dtype=jnp.bfloat16), cpu) for k, v in sorted(peft.items())},
              str(out_dir / "adapter_model.safetensors"))
    cfg = {"alpha_pattern": {}, "auto_mapping": None, "base_model_name_or_path": PEFT_BASE_NAME, "bias": "none",
           "fan_in_fan_out": False, "inference_mode": True, "init_lora_weights": True, "layer_replication": None,
           "layers_pattern": None, "layers_to_transform": sorted(layers), "loftq_config": {},
           "lora_alpha": int(alpha) if float(alpha).is_integer() else alpha, "lora_dropout": 0.0,
           "megatron_config": None, "megatron_core": "megatron.core", "modules_to_save": None, "peft_type": "LORA",
           "r": rank, "rank_pattern": {}, "revision": None,
           "target_modules": target_modules or sorted({k.split(".")[-3] for k in peft}),
           "task_type": "CAUSAL_LM", "use_dora": False, "use_rslora": False}
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    return out_dir


def read_safetensors_f32(path):
    """Read a safetensors file into float32 numpy without torch (bf16 handled through ml_dtypes)."""
    import struct

    import ml_dtypes
    raw = Path(path).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    base = 8 + n
    dt = {"BF16": ml_dtypes.bfloat16, "F32": np.float32, "F16": np.float16}
    out = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        s, e = v["data_offsets"]
        out[k] = np.frombuffer(raw[base + s:base + e], dtype=dt[v["dtype"]]).reshape(v["shape"]).astype(np.float32)
    return out, header


def hf_weight(model_dir, key):
    """One tensor from the HF checkpoint, located through model.safetensors.index.json, as float32."""
    import struct

    import ml_dtypes
    idx = Path(model_dir) / "model.safetensors.index.json"
    fname = json.load(open(idx))["weight_map"][key] if idx.exists() else \
        next(p.name for p in Path(model_dir).glob("*.safetensors"))
    with open(Path(model_dir) / fname, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        v = header[key]
        s, e = v["data_offsets"]
        f.seek(8 + n + s)
        buf = f.read(e - s)
    dt = {"BF16": ml_dtypes.bfloat16, "F32": np.float32, "F16": np.float16}[v["dtype"]]
    return np.frombuffer(buf, dtype=dt).reshape(v["shape"]).astype(np.float32)


def verify_layer(trainer, tensors, saved, L, rank, alpha, seed=0, random_b=False):
    """LoRA forward as Qwix computes it against the PEFT form (x @ W.T + alpha/r * (x @ A.T) @ B.T) built from the
    exported tensors and the HF-layout base weight, for every adapted module of layer L.
    attention (Einsum): base einsum + einsum(lora_einsum_str, x, a, b) * alpha/r
    mlp (nnx.Linear):   x @ kernel + (x @ a) @ b * alpha/r   (Qwix dot_general LoRA, kernel [in, out])"""
    import jax
    scale = alpha / rank
    rng = np.random.default_rng(seed + L)
    layer = trainer.model.layers[L]
    res = {}
    mods = sorted({m for (layer_idx, m, _) in tensors if layer_idx == L})
    for mod in mods:
        parent, name = TUNIX_TO_HF[mod]
        m = getattr(layer.attn if parent == "self_attn" else layer.mlp, mod)
        a = tensors[(L, mod, "a")]
        b = tensors[(L, mod, "b")]
        if random_b:
            b = rng.standard_normal(b.shape).astype(np.float32) * 0.02
            A, B = (to_peft({(L, mod, "a"): a, (L, mod, "b"): b}, rank)[peft_key(L, name, ab)] for ab in "ab")
        else:
            A, B = saved[peft_key(L, name, "a")], saved[peft_key(L, name, "b")]
        lstr = None
        if mod == "q_einsum":
            w = np.asarray(jax.device_get(m.w[...]), np.float32)
            lstr = getattr(m, "w_lora_einsum_str", None)
            N, D, H = w.shape
            x = rng.standard_normal((1, 4, D)).astype(np.float32)
            base_q = np.einsum(m.einsum_str, x, w, optimize=True).reshape(4, N * H)
            delta_q = (np.einsum(lstr, x, a, b, optimize=True) * scale).reshape(4, N * H)
            W_hf = w.transpose(0, 2, 1).reshape(N * H, D)
            xf = x[0]
        elif mod == "attn_vec_einsum":
            w = np.asarray(jax.device_get(m.w[...]), np.float32)
            lstr = getattr(m, "w_lora_einsum_str", None)
            N, H, D = w.shape
            x = rng.standard_normal((1, 4, N, H)).astype(np.float32)
            base_q = np.einsum(m.einsum_str, x, w, optimize=True).reshape(4, D)
            delta_q = (np.einsum(lstr, x, a, b, optimize=True) * scale).reshape(4, D)
            W_hf = w.reshape(N * H, D).T
            xf = x[0].reshape(4, N * H)
        else:
            w = np.asarray(jax.device_get(m.kernel[...]), np.float32)
            n_in, _ = w.shape
            xf = rng.standard_normal((4, n_in)).astype(np.float32)
            base_q = xf @ w
            delta_q = (xf @ a) @ b * scale
            W_hf = w.T
        base_p = xf @ W_hf.T
        delta_p = scale * (xf @ A.T) @ B.T
        full_q, full_p = base_q + delta_q, base_p + delta_p

        def rel(u, v):
            return float(np.abs(u - v).max() / (np.abs(v).max() + 1e-30))

        r = {"einsum": getattr(m, "einsum_str", "linear"), "lora_einsum": lstr, "w_shape": list(w.shape),
             "a_shape": list(a.shape), "b_shape": list(b.shape), "A_shape": list(A.shape), "B_shape": list(B.shape),
             "delta_rel_err": rel(delta_q, delta_p), "full_rel_err": rel(full_q, full_p),
             "base_rel_err": rel(base_q, base_p), "delta_over_base": float(np.abs(delta_p).max() /
                                                                          (np.abs(base_p).max() + 1e-30))}
        if not random_b and MODEL_DIR:
            try:
                hf = hf_weight(MODEL_DIR, f"model.language_model.layers.{L}.{parent}.{name}.weight")
                r["hf_weight_shape"] = list(hf.shape)
                r["hf_weight_equal"] = bool(hf.shape == W_hf.shape and np.array_equal(hf, W_hf))
            except Exception as e:
                r["hf_weight_error"] = str(e)
        res[name] = r
    return res


def _has_dist(md, p):
    try:
        md.version(p)
        return True
    except Exception:
        return False
