"""Fit theta of the latency and energy equations on the calibration grid.

Reads results/measurements.csv and meta.json, writes results/theta.json with errors per split.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls

import equations as eq


def load_measurements(results):
    frame = pd.read_csv(results / "measurements.csv")
    frame["is_validation"] = frame["is_validation"].astype(str).str.lower() == "true"
    frame["is_oom"] = frame["status"] == "OOM"
    frame["memory"] = pd.to_numeric(frame["memory"], errors="coerce")
    return frame


def fit_latency(sizes, batches, latencies, bandwidth):
    """Fit launch_floor and peak_flops on the log ratio (relative error).

    Bandwidth comes from the copy test in measure.py: FLOPs and bytes of this network grow
    together, so both ceilings cannot be recovered from end-to-end timings.
    """

    def to_theta(log_params):
        launch_floor, peak_flops = np.exp(log_params)
        return {"launch_floor": launch_floor, "peak_flops": peak_flops, "bandwidth": bandwidth}

    def residuals(log_params):
        return np.log(eq.latency(sizes, batches, to_theta(log_params))) - np.log(latencies)

    # several starts: max() in the model makes the loss non-convex
    fits = [
        least_squares(residuals, np.log([latencies.min(), peak])) for peak in (1e12, 3e12, 8e12)
    ]
    best = min(fits, key=lambda fit: fit.cost)
    return {key: float(value) for key, value in to_theta(best.x).items()}


def fit_energy(sizes, batches, energies, theta):
    """E = static_power * T + dynamic_power * T_gpu, non-negative least squares on relative error."""
    features = np.column_stack(
        [eq.latency(sizes, batches, theta), eq.gpu_time(sizes, batches, theta)]
    )
    (static_power, dynamic_power), _ = nnls(features / energies[:, None], np.ones_like(energies))
    return {
        "static_power": float(static_power),
        "dynamic_power": float(dynamic_power),
        "latency": theta,
    }


def errors(predicted, measured):
    relative = np.abs(predicted - measured) / measured
    return {
        "n": int(relative.size),
        "mape": float(relative.mean()),
        "median_ape": float(np.median(relative)),
        "max_ape": float(relative.max()),
    }


def errors_per_split(frame, predicted, column):
    measured = frame[column].to_numpy(float)
    is_validation = frame["is_validation"].to_numpy()
    return {
        "calibration": errors(predicted[~is_validation], measured[~is_validation]),
        "validation": errors(predicted[is_validation], measured[is_validation]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    results = parser.parse_args().results

    frame = load_measurements(results)
    meta = json.loads((results / "meta.json").read_text())
    measured = frame[~frame["is_oom"]].reset_index(drop=True)
    sizes, batches = measured["S"].to_numpy(float), measured["B"].to_numpy(float)
    calibration = measured[~measured["is_validation"]]
    cal_sizes, cal_batches = calibration["S"].to_numpy(float), calibration["B"].to_numpy(float)

    bandwidth = meta["copy_bandwidth"]
    theta = fit_latency(cal_sizes, cal_batches, calibration["latency"].to_numpy(float), bandwidth)
    theta_energy = fit_energy(cal_sizes, cal_batches, calibration["energy"].to_numpy(float), theta)

    all_sizes, all_batches = frame["S"].to_numpy(float), frame["B"].to_numpy(float)
    predicted_memory = eq.memory(all_sizes, all_batches)
    metrics = {
        "latency": errors_per_split(measured, eq.latency(sizes, batches, theta), "latency"),
        "energy": errors_per_split(measured, eq.energy(sizes, batches, theta_energy), "energy"),
        "memory": errors_per_split(measured, eq.memory(sizes, batches), "memory"),
        "flops_vs_counter": errors(
            eq.flops(all_sizes, all_batches), frame["flops_counted"].to_numpy(float)
        ),
        "oom": {
            "measured": int(frame["is_oom"].sum()),
            "predicted_over_total_memory": int(
                (predicted_memory > meta["gpu_total_memory_bytes"]).sum()
            ),
        },
    }

    output = {"latency": theta, "energy": theta_energy, "metrics": metrics}
    (results / "theta.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
