import os
import sys
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch


TOTAL_WORKERS = int(os.getenv("TOTAL_WORKERS", "80"))
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
EPOCHS_OVERRIDE = os.getenv("EPOCHS", "10")
MAX_TRAIN_BATCHES = os.getenv("MAX_TRAIN_BATCHES")
AUTO_TUNE_ON_OOM = os.getenv("AUTO_TUNE_ON_OOM", "1") == "1"


def round_down_pow2(x: int) -> int:
    if x < 1:
        return 1
    return 1 << (x.bit_length() - 1)


def estimate_best_batch(script_name: str, vram_gb: float) -> int:
    per_sample_gb = {
        "vitb16_mnist.py": 0.06,
        "vitb16_coco.py": 0.22,
        "resnet101_mnist.py": 0.03,
        "resnet101_coco.py": 0.10,
    }
    max_cap = {
        "vitb16_mnist.py": 1024,
        "vitb16_coco.py": 256,
        "resnet101_mnist.py": 2048,
        "resnet101_coco.py": 512,
    }

    ps = per_sample_gb.get(script_name, 0.12)
    raw = int((vram_gb * 0.9) / ps)
    batch = round_down_pow2(max(1, raw))
    return min(batch, max_cap.get(script_name, 256))


def gpu_vram_gb(gpu_idx: int) -> float:
    props = torch.cuda.get_device_properties(gpu_idx)
    return props.total_memory / (1024 ** 3)


def run_task(script_path: Path, gpu_id: int, workers_per_task: int):
    script_name = script_path.name
    vram = gpu_vram_gb(gpu_id)
    batch_size = estimate_best_batch(script_name, vram)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{script_path.stem}_gpu{gpu_id}.log"

    env = os.environ.copy()
    env["CUDA_DEVICE"] = str(gpu_id)
    env["NUM_WORKERS"] = str(workers_per_task)
    current_batch = batch_size
    if EPOCHS_OVERRIDE:
        env["EPOCHS"] = EPOCHS_OVERRIDE
    if MAX_TRAIN_BATCHES:
        env["MAX_TRAIN_BATCHES"] = MAX_TRAIN_BATCHES

    cmd = [sys.executable, str(script_path)]

    while True:
        env["BATCH_SIZE"] = str(current_batch)
        print(
            f"[GPU {gpu_id}] Starting {script_name} | workers={workers_per_task} "
            f"| batch={current_batch} | vram={vram:.1f}GB | log={log_path}"
        )

        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.Popen(cmd, cwd=str(script_path.parent), env=env, stdout=logf, stderr=subprocess.STDOUT)
            code = proc.wait()

        if code == 0:
            print(f"[GPU {gpu_id}] Finished {script_name} successfully")
            return

        oom_detected = False
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
            if "out of memory" in text or "cuda error: out of memory" in text:
                oom_detected = True
        except Exception:
            pass

        if AUTO_TUNE_ON_OOM and oom_detected and current_batch > 1:
            new_batch = max(1, current_batch // 2)
            if new_batch == current_batch:
                new_batch = max(1, current_batch - 1)
            print(f"[GPU {gpu_id}] OOM in {script_name} at batch={current_batch}; retrying with batch={new_batch}")
            current_batch = new_batch
            continue

        print(f"[GPU {gpu_id}] {script_name} exited with code {code}. Check {log_path}")
        return


def main():
    project_dir = Path(__file__).resolve().parent

    requested_scripts = [
        "vitb16_coco.py",
        "vitb16_mnist.py",
        "resnet101_mnist.py",
        "resnet101_coco.py",
    ]

    scripts = []
    for name in requested_scripts:
        path = project_dir / name
        if path.exists():
            scripts.append(path)
        else:
            print(f"Skipping missing script: {name}")

    if not scripts:
        raise RuntimeError("No target scripts found to run.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This runner expects 2 GPUs.")

    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        raise RuntimeError(f"Expected 2 GPUs, found {gpu_count}.")

    workers_per_gpu = max(1, TOTAL_WORKERS // 2)
    workers_per_task = workers_per_gpu  # one active task per GPU at a time

    # Round-robin assignment so jobs run in parallel across 2 GPUs.
    gpu_queues = {0: [], 1: []}
    for i, script in enumerate(scripts):
        gpu_queues[i % 2].append(script)

    def gpu_worker(gpu_id: int):
        for script in gpu_queues[gpu_id]:
            run_task(script, gpu_id, workers_per_task)

    print(f"Launching merged run with TOTAL_WORKERS={TOTAL_WORKERS} ({workers_per_gpu} per GPU) on 2 GPUs")
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = [ex.submit(gpu_worker, 0), ex.submit(gpu_worker, 1)]
        for fut in futures:
            fut.result()

    print("All scheduled scripts completed.")


if __name__ == "__main__":
    main()
