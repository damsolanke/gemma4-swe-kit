"""LoRA tools without MLX, torch or a model: argument parsing, the trainer's datasets, PEFT key names and config, and
the key check's comparison logic (PEFT's expected tree is stubbed)."""
import json
import struct
import sys

import pytest

from gemma4_swe_kit.lora import check_peft, mlx_to_peft, train_mlx


class WordTokenizer:
    """One token id per whitespace-separated word, ids from a growing vocabulary."""

    def __init__(self):
        self.vocab = {}

    def encode(self, text, add_special_tokens=False):
        return [self.vocab.setdefault(w, len(self.vocab)) for w in text.split()]


def test_train_parser_defaults():
    a = train_mlx.build_parser().parse_args(["--data", "d", "--out", "o"])
    assert (a.base, a.rank, a.scale, a.keys, a.num_layers) == (
        "mlx-community/gemma-4-31B-it-qat-4bit", 16, 2.0, "self_attn.q_proj,self_attn.o_proj", 0)
    assert (a.iters, a.lr, a.max_len, a.val_batches, a.steps_per_eval, a.save_every, a.limit, a.stop_token) == (
        300, 1e-5, 12288, 20, 100, 100, 0, "<|tool_response>")
    with pytest.raises(SystemExit):
        train_mlx.build_parser().parse_args(["--out", "o"])


def test_train_needs_mlx(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    monkeypatch.setitem(sys.modules, "mlx.optimizers", None)
    with pytest.raises(SystemExit, match="needs mlx-lm"):
        train_mlx.main(["--data", str(tmp_path), "--out", str(tmp_path / "out")])


def test_segment_windows(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [{"segments": [["<h> sys user", 0], ["call a", 1], ["result", 0], ["call b <end>", 1]]},
            {"segments": [["only context", 0]]},                                    # no target: skipped
            {"segments": [["w " * 50, 0], ["x", 1]]}]                               # longer than max_len: skipped
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ds = train_mlx.Segments(path, WordTokenizer(), max_len=20)
    assert len(ds) == 1
    toks, mask = ds[0]
    assert len(toks) == 9 and mask == [0, 0, 0, 1, 1, 0, 1, 1, 1]


def test_prerendered_pairs(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [{"prompt": "a b c", "completion": "d e"}, {"prompt": "a " * 30, "completion": "z"}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ds = train_mlx.PreRendered(path, WordTokenizer(), stop_ids=[99], max_len=10)
    assert len(ds) == 1 and ds.process(ds[0]) == ds[0]
    toks, offset = ds[0]
    assert offset == 3 and toks[-1] == 99 and len(toks) == 6


def test_peft_names_and_config():
    name, layer, mod = mlx_to_peft.peft_key("language_model.model.layers.30.self_attn.q_proj.lora_a")
    assert name == "base_model.model.model.language_model.layers.30.self_attn.q_proj.lora_A.weight"
    assert (layer, mod) == (30, "q_proj")
    assert mlx_to_peft.peft_key("model.layers.7.mlp.down_proj.lora_b")[0].endswith("layers.7.mlp.down_proj.lora_B.weight")
    with pytest.raises(SystemExit, match="unexpected key"):
        mlx_to_peft.peft_key("model.embed_tokens.lora_a")
    cfg = mlx_to_peft.peft_config({"rank": 16, "scale": 2.0}, {59, 30, 31}, {"q_proj", "o_proj"})
    assert (cfg["r"], cfg["lora_alpha"], cfg["layers_to_transform"], cfg["target_modules"]) == (
        16, 32, [30, 31, 59], ["o_proj", "q_proj"])
    assert isinstance(cfg["lora_alpha"], int) and cfg["base_model_name_or_path"] == "google/gemma-4-31b-it-qat-w4a16-ct"
    assert mlx_to_peft.peft_config({"rank": 8, "scale": 0.3}, {1}, {"q_proj"})["lora_alpha"] == pytest.approx(2.4)


def test_mlx_adapter_converts_to_peft(tmp_path):
    mx = pytest.importorskip("mlx.core")
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    import numpy as np
    from safetensors.torch import load_file

    src = tmp_path / "mlx"
    src.mkdir()
    a = np.arange(12, dtype=np.float32).reshape(6, 2) / 8          # lora_a [in 6, r 2]
    b = np.arange(8, dtype=np.float32).reshape(2, 4) / 8           # lora_b [r 2, out 4]
    mx.save_safetensors(str(src / "adapters.safetensors"),
                        {"language_model.model.layers.3.self_attn.o_proj.lora_a": mx.array(a),
                         "language_model.model.layers.3.self_attn.o_proj.lora_b": mx.array(b)})
    (src / "adapter_config.json").write_text(json.dumps({"lora_parameters": {"rank": 2, "scale": 2.0}}))
    assert mlx_to_peft.main([str(src), str(tmp_path / "peft")]) == 0
    header = check_peft.read_header(tmp_path / "peft" / "adapter_model.safetensors")
    key = "base_model.model.model.language_model.layers.3.self_attn.o_proj.lora_{}.weight"
    assert header[key.format("A")]["dtype"] == "BF16" and header[key.format("A")]["shape"] == [2, 6]
    assert header[key.format("B")]["shape"] == [4, 2]
    out = load_file(str(tmp_path / "peft" / "adapter_model.safetensors"))
    assert np.array_equal(out[key.format("A")].to(torch.float32).numpy(), a.T)   # exact in bf16 (multiples of 1/8)
    assert np.array_equal(out[key.format("B")].to(torch.float32).numpy(), b.T)
    cfg = json.loads((tmp_path / "peft" / "adapter_config.json").read_text())
    assert (cfg["r"], cfg["lora_alpha"], cfg["target_modules"], cfg["layers_to_transform"]) == (2, 4, ["o_proj"], [3])


def write_adapter(path, keys, r=2, alpha=4, targets=("q_proj", "o_proj"), layers=(1, 2)):
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(json.dumps({"r": r, "lora_alpha": alpha, "target_modules": list(targets),
                                                          "layers_to_transform": list(layers)}))
    header = json.dumps({k: {"dtype": "BF16", "shape": list(shape), "data_offsets": [0, 0]}
                         for k, shape in keys.items()}).encode()
    (path / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)


def fake_tree(config_dir, r, lora_alpha, layers, targets):
    """What PEFT would create on a model with hidden size 8: q_proj 8 -> 16, o_proj 16 -> 8, down_proj 32 -> 8."""
    dims = {"q_proj": (8, 16), "o_proj": (16, 8), "down_proj": (32, 8)}
    out = {}
    for L in layers:
        for t in targets:
            parent = "mlp" if t == "down_proj" else "self_attn"
            n_in, n_out = dims[t]
            base = f"base_model.model.model.language_model.layers.{L}.{parent}.{t}"
            out[f"{base}.lora_A.weight"], out[f"{base}.lora_B.weight"] = (r, n_in), (n_out, r)
    return out


def test_key_check(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(check_peft, "expected_shapes", fake_tree)
    good = fake_tree(None, 2, 4, [1, 2], ["q_proj", "o_proj"])
    write_adapter(tmp_path / "ok", good)
    assert check_peft.main([str(tmp_path / "ok"), "--config-dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip().endswith("OK")
    assert check_peft.main([str(tmp_path / "ok"), "--config-dir", ".", "--expect-targets", "o_proj,q_proj",
                            "--expect-layers", "1-2"]) == 0
    assert "adapter_config matches" in capsys.readouterr().out
    assert check_peft.main([str(tmp_path / "ok"), "--config-dir", ".", "--expect-targets", "q_proj,o_proj,down_proj"]) == 1
    assert "!= expected ['down_proj', 'o_proj', 'q_proj']" in capsys.readouterr().out
    assert check_peft.main([str(tmp_path / "ok"), "--config-dir", ".", "--expect-layers", "0-2"]) == 1
    bad = dict(good)
    bad["base_model.model.model.language_model.layers.1.self_attn.q_proj.lora_A.weight"] = (2, 9)
    bad["base_model.model.model.language_model.layers.5.self_attn.q_proj.lora_A.weight"] = (2, 8)
    write_adapter(tmp_path / "bad", bad)
    assert check_peft.main([str(tmp_path / "bad"), "--config-dir", "."]) == 1
    assert "unexpected 1, shape mismatches 1" in capsys.readouterr().out
    write_adapter(tmp_path / "kv", good, targets=("q_proj", "qkv_proj"))
    with pytest.raises(SystemExit, match="unknown target modules"):
        check_peft.main([str(tmp_path / "kv"), "--config-dir", "."])


def test_key_check_helpers():
    assert check_peft.parse_layers("0,5,30-32") == [0, 5, 30, 31, 32]
    assert check_peft.target_pattern([30, 31], ["q_proj", "down_proj", "o_proj"]) == (
        r"model\.language_model\.layers\.(30|31)\.(self_attn\.(q_proj|o_proj)|mlp\.(down_proj))")
