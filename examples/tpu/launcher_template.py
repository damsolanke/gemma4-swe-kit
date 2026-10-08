"""Kaggle TPU v5e-8 launcher for the Gemma 4 31B distillation LoRA run (train.py with train_lib.py).

Generated file: edit launcher_template.py, train.py and train_lib.py, then run build_kernel.py.
  1. TPU guard in a subprocess with the image's own jax: exit at once unless jax.device_count() == 8.
  2. Fresh Python 3.12 venv (uv) with the pinned lock (jax[tpu] 0.11.0 + libtpu, flax 0.12.8, qwix 0.1.8,
     Tunix e80311f), TPU_LIBRARY_PATH dropped so the venv's libtpu is used.
  3. Runs the embedded trainer (train.py, importing train_lib.py) with an 8.5-hour wall-clock guard measured
     from this script's start; the trainer stops 8 minutes before it to validate and export, this launcher kills it at
     the guard. Trainer settings come from EXTRA_ENV (C2_* and SMOKE_* overrides set at build time).
Inputs (anywhere under /kaggle/input): the Gemma 4 31B checkpoint (model-00001-of-*.safetensors or model.safetensors
next to its config.json and tokenizer.json) and the dataset folder from prep_data.py (manifest.json, found under the
dataset slug given to build_kernel.py).
Outputs: /kaggle/working/ckpt_NNNNN/ and ckpt_final/ (PEFT adapters), /kaggle/working/resume/ (fp32 LoRA + Adam state),
/kaggle/working/train_out/{metrics.json, inner.log, launcher.json, pip_list.txt}
"""
import glob
import json
import os
import signal
import subprocess
import sys
import threading
import time

T0 = time.time()
# The SMOKE_* overrides exist only for local tests of this launcher; Kaggle sets none of them.
GUARD_S = float(os.environ.get("SMOKE_LAUNCHER_GUARD_S", 8.5 * 3600))
WORK = os.environ.get("SMOKE_WORK", "/kaggle/working")
INPUT = os.environ.get("SMOKE_INPUT_ROOT", "/kaggle/input")
VENV = os.environ.get("SMOKE_VENV", "/tmp/tx")
SKIP_GUARD = os.environ.get("SMOKE_SKIP_GUARD") == "1"
OUT = os.path.join(WORK, "train_out")
# build_kernel.py replaces the double-underscore names below with literals
INNER_FILES = __INNER_FILES__  # noqa: F821
INNER_MAIN = "train.py"
DATA_SLUG = __DATA_SLUG__  # noqa: F821
LOCK = __LOCK__  # noqa: F821
if os.environ.get("SMOKE_LOCK_FILE"):
    LOCK = open(os.environ["SMOKE_LOCK_FILE"]).read()
EXTRA_ENV = __EXTRA_ENV__  # noqa: F821
os.makedirs(OUT, exist_ok=True)
STATUS = {"t0": T0, "steps": {}}


def log(msg):
    print(f"[launcher {(time.time() - T0) / 60:6.2f}m] {msg}", flush=True)


def save_status():
    STATUS["elapsed_s"] = time.time() - T0
    with open(os.path.join(OUT, "launcher.json"), "w") as f:
        json.dump(STATUS, f, indent=1, default=str)


def run(cmd, timeout=None, env=None):
    t = time.time()
    p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout, env=env)
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out, time.time() - t


def find_one(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[0] if hits else None


# 0. environment ---------------------------------------------------------------------------------------------
for cmd in ("free -g", "nproc", "df -h /kaggle/working /tmp /dev/shm", "python3 -V", "ls /dev | grep -i -E 'accel|vfio' | head"):
    rc, out, _ = run(cmd)
    log(f"$ {cmd}\n{out.strip()}")
STATUS["env"] = {k: v for k, v in os.environ.items() if any(s in k for s in ("TPU", "XLA", "JAX", "PJRT", "LIBTPU"))}
log(f"env {STATUS['env']}")
model_dir = os.path.dirname(find_one(f"{INPUT}/**/model-00001-of-*.safetensors")
                            or find_one(f"{INPUT}/**/model.safetensors") or "")
data_dir = os.path.dirname((find_one(f"{INPUT}/**/{DATA_SLUG}/**/manifest.json") if DATA_SLUG else None)
                           or find_one(f"{INPUT}/**/*_train.jsonl") or "")
STATUS["model_dir"], STATUS["data_dir"] = model_dir, data_dir
log(f"model_dir {model_dir}\ndata_dir {data_dir}")
if model_dir:
    rc, out, _ = run(f"ls -la {model_dir}")
    log(out)
save_status()

# 1. TPU guard (image jax, separate process so this process never holds the TPU) -------------------------------
rc, out, dt = run([sys.executable, "-c", "import jax; d = jax.devices(); print('DEVCOUNT', len(d), d[0].platform, "
                   "d[0].device_kind, jax.__version__)"], timeout=900)
dev_line = [line for line in out.splitlines() if line.startswith("DEVCOUNT")]
log(f"guard ({dt:.0f}s): {dev_line or out[-2000:]}")
STATUS["steps"]["guard"] = {"rc": rc, "line": dev_line, "s": dt}
ok = bool(dev_line) and dev_line[0].split()[1] == "8" and dev_line[0].split()[2] == "tpu"
if SKIP_GUARD:
    ok = True
if not ok and rc != 0 and "No module named" in out:
    log("image jax not importable; the venv check below decides")
    ok = None
if ok is False:
    STATUS["result"] = "no 8-chip TPU attached: exit"
    save_status()
    log(STATUS["result"])
    sys.exit(0)
save_status()

# 2. venv -------------------------------------------------------------------------------------------------------
t = time.time()
lock_path = os.path.join(os.path.dirname(VENV.rstrip("/")), "smoke_lock.txt")
with open(lock_path, "w") as f:
    f.write(LOCK)
steps = [
    f"{sys.executable} -m pip install -q 'uv==0.12.19'",
    f"{sys.executable} -m uv venv {VENV} --python 3.12 --clear -q",
    f"{sys.executable} -m uv pip install --python {VENV}/bin/python -q --no-deps -r {lock_path}",
]
for cmd in steps:
    rc, out, dt = run(cmd, timeout=1500)
    log(f"$ {cmd} -> rc {rc} ({dt:.0f}s)\n{out.strip()[-3000:]}")
    if rc != 0:
        STATUS["result"] = f"install failed: {cmd}"
        log(STATUS["result"])
        save_status()
        sys.exit(0)
STATUS["steps"]["install_s"] = time.time() - t
rc, out, _ = run(f"{sys.executable} -m uv pip list --python {VENV}/bin/python")
with open(os.path.join(OUT, "pip_list.txt"), "w") as f:
    f.write(out)
save_status()

env = dict(os.environ)
for k in ("TPU_LIBRARY_PATH", "PJRT_DEVICE", "XRT_TPU_CONFIG"):
    env.pop(k, None)
env.update({"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "JAX_PLATFORMS": "tpu,cpu", "SMOKE_T0": str(T0), "SMOKE_GUARD_S": str(GUARD_S),
            "FULL_GUARD_S": str(GUARD_S), "FULL_CKPT_ROOT": WORK,
            "SMOKE_MODEL_DIR": model_dir, "SMOKE_DATA_DIR": data_dir, "SMOKE_OUT_DIR": OUT,
            "XLA_FLAGS": "--xla_llvm_disable_expensive_passes=true", "TF_CPP_MIN_LOG_LEVEL": "1"})
env.update(EXTRA_ENV)
if ok is None:
    rc, out, dt = run([f"{VENV}/bin/python", "-c", "import jax; d = jax.devices(); print('DEVCOUNT', len(d), "
                       "d[0].platform)"], timeout=900, env=env)
    log(f"venv guard: {out.strip()[-500:]}")
    if "DEVCOUNT 8 tpu" not in out:
        STATUS["result"] = "no 8-chip TPU attached (venv check): exit"
        save_status()
        sys.exit(0)

# 3. trainer ------------------------------------------------------------------------------------------------------
for name, src in INNER_FILES.items():
    with open(os.path.join(OUT, name), "w") as f:
        f.write(src)
inner_path = os.path.join(OUT, INNER_MAIN)
log_f = open(os.path.join(OUT, "inner.log"), "w")
p = subprocess.Popen([f"{VENV}/bin/python", "-u", inner_path], env=env, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)


def pump():
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        log_f.write(line)
        log_f.flush()


th = threading.Thread(target=pump, daemon=True)
th.start()
hard = T0 + GUARD_S - 45
while p.poll() is None and time.time() < hard:
    time.sleep(5)
if p.poll() is None:
    log("GUARD: wall clock reached, stopping the trainer")
    STATUS["guard_kill"] = True
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(p.pid, sig)
        except Exception:
            pass
        try:
            p.wait(timeout=10)
            break
        except subprocess.TimeoutExpired:
            continue
th.join(timeout=10)
log_f.close()
STATUS["inner_rc"] = p.returncode
STATUS["result"] = "trainer finished" if p.returncode == 0 else f"trainer rc {p.returncode}"
for root, _, files in os.walk(WORK):
    for fn in files:
        fp = os.path.join(root, fn)
        log(f"output {fp} {os.path.getsize(fp)}")
save_status()
log(f"done: {STATUS['result']}")
