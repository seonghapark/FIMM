"""
How to use:
- From the project root, run: `python3 measure_resources.py`
- Results are saved to: `resource_benchmark_results.csv`
- Requires: PyTorch (and `psutil` optional for CPU/RAM sampling).
"""

import ast
import csv
import inspect
import os
import subprocess
import threading
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import psutil
except Exception:
    psutil = None

TARGET_FILES = [
    "resnet18_mnist_all_final.py",
    "resnet18_coco_all_final.py",
    "vitb16_mnist.py",
    "vitb16_coco.py",
]

OUTPUT_CSV = "resource_benchmark_results.csv"


@dataclass
class SampleStats:
    cpu_percent_peak: float = 0.0
    rss_mb_peak: float = 0.0
    gpu_util_peak: float = 0.0
    gpu_mem_mb_peak: float = 0.0

def load_module_defs_only(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        source = file.read()

    tree = ast.parse(source, filename=file_path)
    def is_safe_constant_expr(node):
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return all(is_safe_constant_expr(elt) for elt in node.elts)
        if isinstance(node, ast.Dict):
            return all(
                (k is None or is_safe_constant_expr(k)) and is_safe_constant_expr(v)
                for k, v in zip(node.keys, node.values)
            )
        if isinstance(node, ast.UnaryOp):
            return is_safe_constant_expr(node.operand)
        if isinstance(node, ast.BinOp):
            return is_safe_constant_expr(node.left) and is_safe_constant_expr(node.right)
        return False

    def is_safe_assign(node):
        if not isinstance(node, ast.Assign):
            return False
        if not all(isinstance(target, ast.Name) and target.id.isupper() for target in node.targets):
            return False
        return is_safe_constant_expr(node.value)

    allowed_nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            allowed_nodes.append(node)
        elif is_safe_assign(node):
            allowed_nodes.append(node)

    filtered_module = ast.Module(body=allowed_nodes, type_ignores=[])
    code_obj = compile(filtered_module, file_path, "exec")
    namespace = {}
    exec(code_obj, namespace, namespace)
    return namespace


def choose_num_classes(file_path):
    return 80 if "coco" in file_path.lower() else 10


def choose_input_shape(file_path):
    low = file_path.lower()
    if "coco" in low:
        return (1, 3, 224, 224)
    if "vit" in low:
        return (1, 1, 224, 224)
    return (2, 1, 28, 28)


def instantiate_model(cls, file_path, namespace):
    sig = inspect.signature(cls)
    kwargs = {}

    if "num_classes" in sig.parameters:
        kwargs["num_classes"] = choose_num_classes(file_path)
    if "k_probes" in sig.parameters:
        default_k = 224 if "coco" in file_path.lower() else 28
        kwargs["k_probes"] = namespace.get("K_PROBES", default_k)
    if "epsilon" in sig.parameters:
        kwargs["epsilon"] = namespace.get("EPSILON", 0.1)

    return cls(**kwargs)


def sample_gpu(gpu_index):
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
        "-i",
        str(gpu_index),
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
        util_str, mem_str = [part.strip() for part in output.split(",")[:2]]
        return float(util_str), float(mem_str)
    except Exception:
        return 0.0, 0.0


def sampling_loop(stop_event, stats, gpu_index):
    proc = psutil.Process(os.getpid()) if psutil else None
    if proc is not None:
        proc.cpu_percent(interval=None)

    while not stop_event.is_set():
        if proc is not None:
            cpu = proc.cpu_percent(interval=0.0)
            rss = proc.memory_info().rss / (1024 ** 2)
            stats.cpu_percent_peak = max(stats.cpu_percent_peak, cpu)
            stats.rss_mb_peak = max(stats.rss_mb_peak, rss)

        if gpu_index is not None:
            util, mem = sample_gpu(gpu_index)
            stats.gpu_util_peak = max(stats.gpu_util_peak, util)
            stats.gpu_mem_mb_peak = max(stats.gpu_mem_mb_peak, mem)

        time.sleep(0.2)


def benchmark_model(model, file_path, class_name, device):
    input_shape = choose_input_shape(file_path)
    num_classes = choose_num_classes(file_path)

    x = torch.randn(*input_shape, device=device)
    y = torch.randint(0, num_classes, (input_shape[0],), device=device)

    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    gpu_index = torch.cuda.current_device() if device.type == "cuda" else None
    stats = SampleStats()
    stop_event = threading.Event()
    sampler = threading.Thread(target=sampling_loop, args=(stop_event, stats, gpu_index), daemon=True)
    sampler.start()

    step_times = []
    error = ""

    try:
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        for _ in range(3):
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_times.append(time.perf_counter() - start)

    except Exception as ex:
        error = str(ex)

    finally:
        stop_event.set()
        sampler.join(timeout=1.0)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    avg_step_ms = (sum(step_times) / len(step_times) * 1000.0) if step_times else 0.0
    peak_gpu_mem_torch_mb = 0.0
    if device.type == "cuda":
        peak_gpu_mem_torch_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        "file": file_path,
        "class": class_name,
        "device": str(device),
        "avg_step_ms": round(avg_step_ms, 3),
        "cpu_percent_peak": round(stats.cpu_percent_peak, 2),
        "rss_mb_peak": round(stats.rss_mb_peak, 2),
        "gpu_util_peak": round(stats.gpu_util_peak, 2),
        "gpu_mem_mb_peak_nvsmi": round(stats.gpu_mem_mb_peak, 2),
        "gpu_mem_mb_peak_torch": round(peak_gpu_mem_torch_mb, 2),
        "error": error,
    }


def iter_classifier_classes(namespace):
    for name, obj in namespace.items():
        if not isinstance(obj, type):
            continue
        if name.endswith("NFViT"):
            continue
        try:
            if issubclass(obj, nn.Module) and "Classifier" in name:
                yield name, obj
        except Exception:
            continue


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass

    rows = []
    files = TARGET_FILES

    print("Benchmark target files:")
    for path in files:
        print(f" - {path}")

    for file_path in files:
        print(f"\n[Load] {file_path}")
        try:
            namespace = load_module_defs_only(file_path)
        except Exception as ex:
            rows.append({
                "file": file_path,
                "class": "<load_error>",
                "device": str(device),
                "avg_step_ms": 0.0,
                "cpu_percent_peak": 0.0,
                "rss_mb_peak": 0.0,
                "gpu_util_peak": 0.0,
                "gpu_mem_mb_peak_nvsmi": 0.0,
                "gpu_mem_mb_peak_torch": 0.0,
                "error": f"Load failed: {ex}",
            })
            continue

        class_items = sorted(iter_classifier_classes(namespace), key=lambda item: item[0])
        for class_name, cls in class_items:
            print(f"[Run] {file_path} :: {class_name}")
            try:
                model = instantiate_model(cls, file_path, namespace)
                result = benchmark_model(model, file_path, class_name, device)
            except Exception as ex:
                result = {
                    "file": file_path,
                    "class": class_name,
                    "device": str(device),
                    "avg_step_ms": 0.0,
                    "cpu_percent_peak": 0.0,
                    "rss_mb_peak": 0.0,
                    "gpu_util_peak": 0.0,
                    "gpu_mem_mb_peak_nvsmi": 0.0,
                    "gpu_mem_mb_peak_torch": 0.0,
                    "error": f"Setup/benchmark failed: {ex}",
                }
            rows.append(result)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved benchmark results to {OUTPUT_CSV}")
    for row in rows:
        status = "OK" if not row["error"] else f"ERR: {row['error'][:80]}"
        print(
            f"{row['file']} | {row['class']} | {row['avg_step_ms']} ms/step | "
            f"CPU% peak {row['cpu_percent_peak']} | RAM {row['rss_mb_peak']} MB | "
            f"GPU util {row['gpu_util_peak']}% | GPU mem {row['gpu_mem_mb_peak_torch']} MB | {status}"
        )


if __name__ == "__main__":
    main()
