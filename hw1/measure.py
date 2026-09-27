import argparse
import csv
import json
import platform
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import pynvml
import torch
from torch.utils.flop_counter import FlopCounterMode

from models import build_model

BASE_SIZES = (32, 64, 128, 224, 256, 384, 512)
BASE_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
SEED = 2026


def sample_grid(seed):
    """Validation grid: 4 sizes (multiples of 16 not in the base grid), 3 non-power-of-two batches."""
    rng = np.random.default_rng(seed)
    size_pool = [s for s in range(32, 513, 16) if s not in BASE_SIZES]
    batch_pool = [b for b in range(1, 257) if b & (b - 1) != 0]
    sizes = sorted(int(s) for s in rng.choice(size_pool, 4, replace=False))
    batches = sorted(int(b) for b in rng.choice(batch_pool, 3, replace=False))
    return sizes, batches


def timings(fn, min_calls, min_seconds=0.0, max_calls=200):
    """Durations of single synchronized calls of fn."""
    durations = []
    while len(durations) < max_calls and (
        len(durations) < min_calls or sum(durations) < min_seconds
    ):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
    return durations


def copy_bandwidth():
    """Real memory bandwidth, bytes/s: copy 256 MB inside the GPU (read + write, no compute)."""
    copy_bytes = 256 * 2**20
    source = torch.empty(copy_bytes // 4, device="cuda")
    target = torch.empty_like(source)
    target.copy_(source)  # warm-up
    seconds = statistics.median(timings(lambda: target.copy_(source), 20, max_calls=20))
    return 2 * copy_bytes / seconds


def has_energy_counter(gpu):
    try:
        pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu)
        return True
    except pynvml.NVMLError:
        return False


def energy_per_call(gpu, use_counter, fn, min_seconds=2.0, min_calls=3):
    """Whole-GPU joules per call while fn runs back to back for at least min_seconds."""
    samples = []  # (time, watts), only when there is no energy counter
    stop = threading.Event()

    def sample_power():
        while not stop.is_set():
            samples.append((time.perf_counter(), pynvml.nvmlDeviceGetPowerUsage(gpu) / 1000))
            time.sleep(0.005)

    torch.cuda.synchronize()
    if use_counter:
        start_joules = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu) / 1000
    else:
        thread = threading.Thread(target=sample_power, daemon=True)
        thread.start()

    calls = 0
    start = time.perf_counter()
    while calls < min_calls or time.perf_counter() - start < min_seconds:
        fn()
        calls += 1
        if calls % 8 == 0:  # keep the queue short so the host clock tracks the GPU
            torch.cuda.synchronize()
    torch.cuda.synchronize()

    if use_counter:
        joules = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu) / 1000 - start_joules
    else:
        stop.set()
        thread.join()
        times, watts = np.array(samples).T
        joules = np.sum((watts[1:] + watts[:-1]) / 2 * np.diff(times))  # trapezoid rule
    return float(joules) / calls


def count_flops(model_on_meta, image_size, batch):
    inputs = torch.empty(batch, 3, image_size, image_size, device="meta")
    with FlopCounterMode(display=False) as counter:
        model_on_meta(inputs)
    return counter.get_total_flops()


def measure_config(model, gpu, use_counter, image_size, batch):
    inputs = torch.randn(batch, 3, image_size, image_size, device="cuda")
    for _ in range(3):
        model(inputs)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    model(inputs)
    torch.cuda.synchronize()
    memory = torch.cuda.max_memory_allocated()

    latency = statistics.median(timings(lambda: model(inputs), min_calls=5, min_seconds=0.5))
    energy = energy_per_call(gpu, use_counter, lambda: model(inputs))
    return {"status": "ok", "latency": latency, "memory": memory, "energy": energy}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results"))
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    use_counter = has_energy_counter(gpu)

    random_sizes, random_batches = sample_grid(SEED)
    sizes = sorted(set(BASE_SIZES) | set(random_sizes))
    batches = sorted(set(BASE_BATCHES) | set(random_batches))
    configs = [(s, b) for s in sizes for b in batches]
    # random order decorrelates thermal drift from (S, B)
    order = np.random.default_rng(SEED).permutation(len(configs))

    model = build_model().cuda().eval()
    model_on_meta = build_model().to("meta").eval()

    with torch.inference_mode():
        # 20 s of work brings clocks and temperature to a steady state
        warmup_inputs = torch.randn(64, 3, 224, 224, device="cuda")
        start = time.perf_counter()
        while time.perf_counter() - start < 20:
            model(warmup_inputs)
            torch.cuda.synchronize()
        del warmup_inputs

        properties = torch.cuda.get_device_properties(0)
        meta = {
            "gpu": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
            "driver": pynvml.nvmlSystemGetDriverVersion(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "python": platform.python_version(),
            "energy_mode": "energy_counter" if use_counter else "power_sampling",
            "seed": SEED,
            "random_sizes": random_sizes,
            "random_batches": random_batches,
            "copy_bandwidth": copy_bandwidth(),
        }
        torch.cuda.empty_cache()  # release the copy buffers before measuring memory
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))

        rows = []
        for step, index in enumerate(order, start=1):
            image_size, batch = configs[index]
            row = {
                "S": image_size,
                "B": batch,
                "is_validation": image_size in random_sizes or batch in random_batches,
                "flops_counted": count_flops(model_on_meta, image_size, batch),
            }
            try:
                row |= measure_config(model, gpu, use_counter, image_size, batch)
            except torch.cuda.OutOfMemoryError:
                row |= {"status": "OOM", "memory": "OOM"}
            torch.cuda.empty_cache()
            rows.append(row)
            print(f"[{step}/{len(configs)}] {row}")

    rows.sort(key=lambda row: (row["S"], row["B"]))
    columns = ["S", "B", "latency", "memory", "energy", "is_validation", "status", "flops_counted"]
    with open(out / "measurements.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out / 'measurements.csv'}")


if __name__ == "__main__":
    main()
