"""g4kit-check-peft: check a PEFT adapter against the parameter names and shapes PEFT itself creates on the Gemma 4
module tree (built on the meta device from config.json, no weights loaded).

The expected set is built for exactly the (layer, module) pairs listed in adapter_config.json: target_modules under
self_attn (q_proj, k_proj, v_proj, o_proj) or mlp (gate_proj, up_proj, down_proj) of the language model's layers in
layers_to_transform (all layers present in the file if absent). Vision-tower modules never match.
  --expect-targets  PEFT module names the adapter must target, exactly (order free). A trainer that falls back to a
                    subset of its targets (examples/tpu) reports the configuration it trained; pass that here.
                    adapter_config.json target_modules must equal it and the expected key set is built from it.
  --expect-layers   layers the adapter must cover, exactly ('30-59', '0,5,30-59'); layers_to_transform (or, if absent,
                    the layers present in the file) must equal it and the expected key set is built from it.
Without the two options the expected set comes from adapter_config.json alone. Prints OK (exit status 0) or MISMATCH
(exit status 1). Needs torch, peft and a transformers release with Gemma 4 (pip install '.[peft]').
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

PARENT = {"q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn", "o_proj": "self_attn",
          "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}


def read_header(path: str | Path) -> dict:
    """The safetensors header (tensor name -> {dtype, shape, data_offsets}) without loading any tensor."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def parse_layers(spec: str) -> list[int]:
    """'30-59' or '0,5,30-59' -> sorted layer numbers."""
    out: set[int] = set()
    for part in (x.strip() for x in spec.split(",") if x.strip()):
        lo, _, hi = part.partition("-")
        out |= set(range(int(lo), int(hi or lo) + 1))
    return sorted(out)


def target_pattern(layers: list[int], targets: list[str]) -> str:
    """PEFT target_modules regex for the given language-model layers and module names."""
    groups = []
    for parent in ("self_attn", "mlp"):
        mods = [t for t in targets if PARENT[t] == parent]
        if mods:
            groups.append(parent + r"\.(" + "|".join(mods) + ")")
    return r"model\.language_model\.layers\.(" + "|".join(map(str, layers)) + r")\.(" + "|".join(groups) + ")"


def expected_shapes(config_dir: Path, r: int, lora_alpha: float, layers: list[int], targets: list[str]) -> dict:
    """{PEFT tensor name: shape} that PEFT creates for these layers and modules on the meta-device model."""
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoConfig, AutoModelForImageTextToText
    except ImportError as exc:
        raise SystemExit(f"g4kit-check-peft needs torch, transformers and peft (pip install '.[peft]'): {exc}") from None
    cfg_path = config_dir if config_dir.is_file() else config_dir / "config.json"
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    raw.pop("quantization", None)
    raw.pop("quantization_config", None)
    config = AutoConfig.for_model(**raw) if "model_type" in raw else AutoConfig.from_pretrained(cfg_path.parent)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config)
    peft_model = get_peft_model(model, LoraConfig(r=r, lora_alpha=lora_alpha, target_modules=target_pattern(layers, targets)))
    return {name.replace(".default", ""): tuple(p.shape) for name, p in peft_model.named_parameters() if "lora_" in name}


def check(adapter_dir: Path, config_dir: Path, expect_targets: str | None = None,
          expect_layers: str | None = None) -> bool:
    acfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    header = read_header(adapter_dir / "adapter_model.safetensors")
    ours = {k: tuple(v["shape"]) for k, v in header.items() if k != "__metadata__"}
    dtypes = sorted({v["dtype"] for k, v in header.items() if k != "__metadata__"})
    layers = acfg.get("layers_to_transform") or sorted({int(re.search(r"layers\.(\d+)\.", k).group(1)) for k in ours})
    targets = acfg["target_modules"]
    unknown = [t for t in targets if t not in PARENT]
    if unknown:
        raise SystemExit(f"unknown target modules {unknown}")
    problems = []
    if expect_targets:
        exp_t = sorted({t.strip() for t in expect_targets.split(",") if t.strip()})
        if not exp_t or not all(t in PARENT for t in exp_t):
            raise SystemExit(f"bad --expect-targets {expect_targets}")
        if sorted(set(targets)) != exp_t:
            problems.append(f"target_modules {sorted(set(targets))} != expected {exp_t}")
        targets = exp_t
    if expect_layers:
        exp_l = parse_layers(expect_layers)
        if sorted(layers) != exp_l:
            problems.append(f"layers {sorted(layers)[:3]}..{sorted(layers)[-3:]} ({len(layers)}) != expected "
                            f"{exp_l[:3]}..{exp_l[-3:]} ({len(exp_l)})")
        layers = exp_l
    expected = expected_shapes(config_dir, acfg["r"], acfg["lora_alpha"], layers, targets)
    missing = sorted(set(expected) - set(ours))
    extra = sorted(set(ours) - set(expected))
    shape_bad = sorted(k for k in set(ours) & set(expected) if ours[k] != expected[k])
    per_module: dict[str, int] = {}
    for k in ours:
        m = re.search(r"\.(self_attn|mlp)\.(\w+)\.lora_", k)
        if m:
            per_module[m.group(2)] = per_module.get(m.group(2), 0) + 1
    print(f"adapter tensors {len(ours)} ({dtypes}, r {acfg['r']}, alpha {acfg['lora_alpha']}, {len(layers)} layers, "
          f"per module {dict(sorted(per_module.items()))}), PEFT-expected {len(expected)}, missing {len(missing)}, "
          f"unexpected {len(extra)}, shape mismatches {len(shape_bad)}")
    for k in (missing[:3] + extra[:3] + shape_bad[:3]):
        print("  ", k, ours.get(k), expected.get(k))
    if expect_targets or expect_layers:
        print(f"expected configuration: targets {sorted(set(targets))}, layers {layers[0]}-{layers[-1]} ({len(layers)}); "
              + ("adapter_config matches" if not problems else "; ".join(problems)))
    ok = not (missing or extra or shape_bad or problems)
    print("OK" if ok else "MISMATCH")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="g4kit-check-peft", description=__doc__.split("\n\n")[0])
    ap.add_argument("adapter_dir", help="PEFT adapter directory (adapter_model.safetensors, adapter_config.json)")
    ap.add_argument("--config-dir", required=True,
                    help="Gemma 4 model directory (or its config.json) that defines the module tree, e.g. a download "
                         "of config.json from google/gemma-4-31B-it-qat-w4a16-ct")
    ap.add_argument("--expect-targets", help="comma list of PEFT module names, e.g. q_proj,o_proj,down_proj")
    ap.add_argument("--expect-layers", help="layer list, e.g. 30-59")
    a = ap.parse_args(argv)
    return 0 if check(Path(a.adapter_dir), Path(a.config_dir), a.expect_targets, a.expect_layers) else 1


if __name__ == "__main__":
    sys.exit(main())
