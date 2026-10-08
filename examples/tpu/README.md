# TPU LoRA trainer (Kaggle TPU v5e-8)

LoRA fine-tuning of Gemma 4 31B on a free Kaggle TPU v5e-8 with Tunix (model) and Qwix (LoRA), exported as a PEFT adapter with the key names the scorer's vLLM loads. It runs as a Kaggle script kernel, so it ships as an example rather than a kit command.

| File | Role |
|---|---|
| `train.py` | The run: tries LoRA configurations until one fits in HBM, sweeps the learning rate, trains on a cosine schedule that ends inside the session, writes checkpoints and the final adapter |
| `train_lib.py` | Tunix model from the HF checkpoint, Qwix LoRA, compiled train and eval steps with a chunked loss, out-of-memory detection, PEFT export and a per-layer check of the export |
| `launcher_template.py` | Kernel entry point: checks for 8 TPU chips, builds a Python 3.12 venv from the lock, runs `train.py` under an 8.5-hour wall-clock guard |
| `build_kernel.py` | Writes `kernel.py` (the launcher with both trainer files and the lock embedded) and `kernel-metadata.json` |
| `prep_data.py` | Builds the dataset folder: the renders, `manifest.json` with validation rows and token hashes, `dataset-metadata.json` |
| `requirements.in`, `requirements.lock.txt` | Pinned stack: jax[tpu] 0.11.0 with libtpu 0.0.44.1, flax 0.12.8, qwix 0.1.8, Tunix at commit e80311f, optax 0.2.8 |

## Inputs

| Input | Source |
|---|---|
| Model | Kaggle model `google/gemma-4/transformers/gemma-4-31b-it-qat-q4_0-unquantized/2` (default of `--model`): the QAT checkpoint in BF16 with `config.json` and `tokenizer.json` |
| Training windows | `g4kit-curate`, then `g4kit-render-distill --encoding single --mask-error-turns tool` twice: `--max-len 12288` (primary) and `--max-len 8192` (fallback after an out-of-memory error) |
| Dataset folder | `prep_data.py` over the two renders, uploaded as a private Kaggle dataset |
| Tokenizer for `prep_data.py` | The Kaggle model's `tokenizer.json`; the copy in `mlx-community/gemma-4-31B-it-qat-4bit` had the same sha256 (`cc8d3a0c...`) |
| Quota | Kaggle grants 20 TPU hours a week; a session is charged its wall time |

## Run

```bash
g4kit-curate data/conv.jsonl data/curated.jsonl --source data/openhands
for len in 12288 8192; do
  g4kit-render-distill data/curated.jsonl data/r_think_$len --mode think --encoding single --mask-error-turns tool \
      --max-len $len --system-prompt submission/prompts/system.md --user-template data/user_template.json \
      --tools-from-log runs/proxy_full.jsonl --tokenizer mlx-community/gemma-4-31B-it-qat-4bit
done
hf download mlx-community/gemma-4-31B-it-qat-4bit tokenizer.json --local-dir assets
python examples/tpu/prep_data.py data/tpu_ds --tokenizer assets/tokenizer.json --dataset-id USER/gemma4-distill \
    --variant think=data/r_think_12288 --variant think8k=data/r_think_8192 \
    --curation-summary data/curated.summary.json
kaggle datasets create -p data/tpu_ds
python examples/tpu/build_kernel.py --out-dir build/tpu --kernel-id USER/gemma4-tpu-lora --dataset USER/gemma4-distill
kaggle kernels push -p build/tpu
```

Settings are environment variables that `build_kernel.py` bakes into the kernel (`KEY=VALUE` after the options). Their `SMOKE_`, `FULL_` and `C2_` prefixes name the runs that introduced them.

| Variable | Default | Effect |
|---|---|---|
| `C2_MODE` | `think` | Renders to train on: `think` and `think8k`, or `nothink` and `nothink8k` (prepare those variants instead) |
| `C2_CONFIGS` | `a` to `f` below | LoRA configurations tried in order, `label:targets@layers@windows; ...` |
| `SMOKE_RANK`, `SMOKE_ALPHA` | 32, 32 | LoRA rank and alpha (scale alpha / r) |
| `C2_EPOCHS` | 2 | Passes over the windows, each in a new seeded order |
| `C2_LR`, `C2_WARMUP`, `C2_END_LR_FRAC` | 2e-4, 30, 0.1 | Peak learning rate, warmup steps, end of the cosine as a fraction of the peak |
| `C2_SWEEP`, `C2_SWEEP_LRS`, `C2_SWEEP_STEPS` | `auto`, `2e-4,1e-4,4e-4`, 300 | Each rate trains the first 300 steps from the same start and the best validation loss continues; `auto` runs the sweep only if the rest of the plan still fits |
| `C2_MAX_VAL` | 64 | Held-out windows for validation |
| `FULL_EVERY` | 1000 | Windows between checkpoints |
| `C2_RESERVE_S` | 480 | Seconds kept before the guard for the final validation and export |

| Configuration | Targets | Layers | Windows |
|---|---|---|---|
| a | q, o, gate, up, down | 30 to 59 | up to 12,288 tokens (buckets 8,192 and 12,288) |
| b | q, o, gate, up, down | 30 to 59 | up to 8,192 |
| c | q, o, down | 30 to 59 | up to 12,288 |
| d | q, o, down | 30 to 59 | up to 8,192 |
| e | q, o | 30 to 59 | up to 12,288 |
| f | q, o | 30 to 59 | up to 8,192 |

k and v are left out on purpose: Tunix fuses them, and on the 10 global layers vLLM loads the K weights into both the K and V slots, so a k_proj adapter would train one function and serve another.

## Outputs

| Path under `/kaggle/working` | Content |
|---|---|
| `ckpt_NNNNN/`, `ckpt_final/` | PEFT adapters (BF16 `adapter_model.safetensors`, `adapter_config.json`) every 1,000 windows and at the end; `target_modules` and `layers_to_transform` describe the configuration that trained |
| `resume/` | fp32 LoRA factors and Adam state of the last checkpoint |
| `train_out/metrics.json` | Every configuration's probe with its HLO temporaries, `chosen_config`, the sweep, per-step losses and times, checkpoints and export checks |
| `train_out/inner.log`, `launcher.json`, `pip_list.txt` | Trainer log, launcher status, installed packages |

Before serving an adapter, check its keys against the targets and layers in `metrics.json` `chosen_config`, then run the logit canary on a vLLM server that serves it:

```bash
hf download mlx-community/gemma-4-31B-it-qat-4bit config.json --local-dir assets/gemma4-config
g4kit-check-peft out/ckpt_final --config-dir assets/gemma4-config --expect-targets down_proj,o_proj,q_proj --expect-layers 30-59
g4kit-canary runs/proxy_full.jsonl --limit 2 --api-base http://127.0.0.1:8000/v1
```

## Measured on Kaggle (October 2026)

| Run | LoRA | Data | Step of 2 windows | Training | Validation loss |
|---|---|---|---|---|---|
| Smoke test (earlier version of this trainer) | rank 16 on q and o, layers 30 to 59 | 64 windows | 4.00 s at 12,288 tokens, 2.66 s at 8,192 | 30 steps | 0.604 to 0.321 on 24 windows |
| Full run (earlier version) | rank 16 on q and o, layers 30 to 59 | 12,156 windows, 1 epoch | 4.00 s and 2.66 s (medians) | 6,078 steps in 6.43 h | 0.604 to 0.222 after 1,000 windows, 0.188 at the end |
| Curated run (this trainer, default list) | rank 32 on q, o and down, layers 30 to 59 (configuration d) | 7,954 windows of up to 8,192 tokens, 2 epochs | 2.80 s (median) | 7,954 steps in 6.15 h | 0.958 to 0.355 after 1,000 windows, 0.301 at the end on 64 windows |

| Cost | Measured |
|---|---|
| Loading the 62.5 GB checkpoint | 469 to 1,761 s |
| Applying a LoRA configuration | 18 to 156 s |
| Probing one configuration (compile, one zero step) | 59 to 200 s; in the curated run a, b and c needed 16.79G, 16.13G and 16.15G of HLO temporaries against 15.75G of HBM per chip, d needed 8.67G |
| One checkpoint (rank 32, 180 tensors) | about 43 s |
| Whole curated session | 7.2 h |
| Queue before a session started | 8 to 11 h |
| One 12,288-token window with `g4kit-train-mlx` on an M4 Max | 200 to 230 s, against about 2 s here |

`C2_FAKE_OOM` (see `train_lib.py`) makes chosen compiles or probe steps fail like an out-of-memory error, so the fallback can be exercised on CPU with a tiny checkpoint. The kit's test suite builds the kernel and checks how the launcher finds the model and the dataset.
