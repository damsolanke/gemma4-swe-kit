"""g4kit-mlx-to-peft: convert an mlx_lm LoRA adapter (adapters.safetensors + adapter_config.json, e.g. from
g4kit-train-mlx) into a PEFT adapter for the scorer's vLLM.

mlx_lm LoRALinear computes y = W x + scale * (x @ lora_a) @ lora_b with lora_a [in, r] and lora_b [r, out].
PEFT computes y = W x + (alpha / r) * B (A x) with lora_A [r, in] and lora_B [out, r].
So lora_A = lora_a.T, lora_B = lora_b.T and lora_alpha = scale * r. Tensors are written in BF16 under the key names
of the competition's sample adapter:
  base_model.model.model.language_model.layers.{L}.{self_attn,mlp}.{module}.lora_{A,B}.weight
Needs mlx, numpy, torch and safetensors (pip install '.[mlx,peft]').
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PEFT_BASE_NAME = "google/gemma-4-31b-it-qat-w4a16-ct"
MLX_KEY = re.compile(r"(?:language_model\.)?model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.lora_([ab])$")


def peft_key(mlx_key: str) -> tuple[str, int, str]:
    """(PEFT tensor name, layer, module) for an mlx_lm LoRA parameter name."""
    m = MLX_KEY.match(mlx_key)
    if not m:
        raise SystemExit(f"unexpected key {mlx_key}")
    layer, block, mod, ab = m.groups()
    return f"base_model.model.model.language_model.layers.{layer}.{block}.{mod}.lora_{ab.upper()}.weight", int(layer), mod


def peft_config(lora_parameters: dict, layers: set[int], modules: set[str], base_model: str = PEFT_BASE_NAME) -> dict:
    lp = lora_parameters
    alpha = lp["scale"] * lp["rank"]
    return {
        "alpha_pattern": {}, "auto_mapping": None, "base_model_name_or_path": base_model,
        "bias": "none", "fan_in_fan_out": False, "inference_mode": True, "init_lora_weights": True,
        "layer_replication": None, "layers_pattern": None, "layers_to_transform": sorted(layers), "loftq_config": {},
        "lora_alpha": int(alpha) if float(alpha).is_integer() else alpha, "lora_dropout": 0.0, "megatron_config": None,
        "megatron_core": "megatron.core", "modules_to_save": None, "peft_type": "LORA", "r": lp["rank"],
        "rank_pattern": {}, "revision": None, "target_modules": sorted(modules), "task_type": "CAUSAL_LM",
        "use_dora": False, "use_rslora": False,
    }


def convert(src: Path, dst: Path, base_model: str = PEFT_BASE_NAME) -> dict:
    try:
        import mlx.core as mx
        import numpy as np
        import torch
        from safetensors.torch import save_file
    except ImportError as exc:
        raise SystemExit(f"g4kit-mlx-to-peft needs mlx, numpy, torch and safetensors (pip install '.[mlx,peft]'): "
                         f"{exc}") from None
    cfg = json.loads((src / "adapter_config.json").read_text(encoding="utf-8"))
    lp = cfg["lora_parameters"]
    w = mx.load(str(src / "adapters.safetensors"))
    out, modules, layers = {}, set(), set()
    for k, v in w.items():
        name, layer, mod = peft_key(k)
        modules.add(mod)
        layers.add(layer)
        out[name] = np.array(v.astype(mx.float32)).T.astype(np.float32)   # [in, r] -> [r, in] ; [r, out] -> [out, r]
    dst.mkdir(parents=True, exist_ok=True)
    save_file({k: torch.from_numpy(v).to(torch.bfloat16).contiguous() for k, v in out.items()},
              str(dst / "adapter_model.safetensors"))
    peft_cfg = peft_config(lp, layers, modules, base_model)
    (dst / "adapter_config.json").write_text(json.dumps(peft_cfg, indent=2))
    print(f"{len(out)} tensors, modules {sorted(modules)}, r={lp['rank']} alpha={peft_cfg['lora_alpha']} -> {dst}")
    return peft_cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="g4kit-mlx-to-peft", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("mlx_adapter", help="mlx_lm adapter directory (adapters.safetensors, adapter_config.json)")
    ap.add_argument("out", help="PEFT adapter directory to write (adapter_model.safetensors, adapter_config.json)")
    ap.add_argument("--base-model", default=PEFT_BASE_NAME, help="base_model_name_or_path written to adapter_config.json")
    a = ap.parse_args(argv)
    convert(Path(a.mlx_adapter), Path(a.out), a.base_model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
