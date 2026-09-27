import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import equations as eq
from calibrate import load_measurements

AXIS_LABELS = {"S": "image size S (px)", "B": "batch size B"}


def draw_points(ax, x, y, color, is_validation):
    ax.scatter(x[~is_validation], y[~is_validation], color=color, s=25, zorder=3)
    ax.scatter(
        x[is_validation], y[is_validation], facecolor="white", edgecolor=color, s=35, zorder=3
    )


def add_marker_legend_entries(ax):
    ax.scatter([], [], color="gray", label="measured, calibration")
    ax.scatter([], [], facecolor="white", edgecolor="gray", label="measured, validation")


def family_figure(frame, column, predict, ylabel, title, x_axes=("S",), scale=1.0):
    """For each x axis: one predicted curve per value of the other variable, measured points on top."""
    fig, axes = plt.subplots(1, len(x_axes), figsize=(8 * len(x_axes), 5), squeeze=False)
    for ax, x in zip(axes[0], x_axes):
        series = "B" if x == "S" else "S"
        values = sorted(frame[series].unique())
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(values)))
        x_curve = np.geomspace(frame[x].min(), frame[x].max(), 200)
        for value, color in zip(values, colors):
            s, b = (x_curve, value) if x == "S" else (value, x_curve)
            ax.plot(x_curve, predict(s, b) * scale, color=color, label=f"{series} = {value}")
            points = frame[(frame[series] == value) & frame[column].notna()]
            draw_points(ax, points[x], points[column] * scale, color, points["is_validation"])
        add_marker_legend_entries(ax)
        ax.set(
            yscale="log",
            xlabel=AXIS_LABELS[x],
            ylabel=ylabel,
            title=f"{title} vs {x} (lines: equation)",
        )
        ax.set_xscale("log", base=2)
    return fig, axes[0]


def parity_figure(frame, column, predicted, label):
    measured = frame[column].to_numpy(float)
    fig, ax = plt.subplots(figsize=(6, 5))
    draw_points(ax, measured, predicted, "royalblue", frame["is_validation"].to_numpy())
    add_marker_legend_entries(ax)
    low = min(measured.min(), predicted.min()) * 0.8
    high = max(measured.max(), predicted.max()) * 1.25
    ax.plot([low, high], [low, high], color="gray", label="predicted = measured")
    ax.plot([low, high], [0.8 * low, 0.8 * high], "--", color="lightgray", label="-20 % / +25 %")
    ax.plot([low, high], [1.25 * low, 1.25 * high], "--", color="lightgray")
    ax.set(xscale="log", yscale="log", xlabel=f"measured {label}", ylabel=f"predicted {label}")
    ax.set_title(f"{label}: predicted vs measured")
    return fig


def save(fig, path):
    """Put a legend right of every axes that has labels and no legend yet, then write the PNG."""
    for ax in fig.axes:
        if ax.get_legend() is None and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    results = parser.parse_args().results
    out = results / "figures"
    out.mkdir(parents=True, exist_ok=True)

    frame = load_measurements(results)
    fitted = json.loads((results / "theta.json").read_text())
    meta = json.loads((results / "meta.json").read_text())
    theta, theta_energy = fitted["latency"], fitted["energy"]
    measured = frame[~frame["is_oom"]].reset_index(drop=True)
    sizes, batches = measured["S"].to_numpy(float), measured["B"].to_numpy(float)

    # FlopCounterMode does not count the bias adds of the linear layers
    fig, _ = family_figure(
        frame,
        "flops_counted",
        lambda s, b: eq.flops(s, b) - (eq.HEAD_HIDDEN + eq.NUM_CLASSES) * b,
        "GFLOPs per forward pass",
        "FLOPs (dots: FlopCounterMode)",
        scale=1e-9,
    )
    save(fig, out / "flops.png")

    fig, (ax,) = family_figure(
        frame, "memory", eq.memory, "peak allocated memory (MiB)", "Memory", scale=2**-20
    )
    ax.axhline(
        meta["gpu_total_memory_bytes"] / 2**20, color="red", linestyle="--", label="device memory"
    )
    oom = frame[frame["is_oom"]]
    oom_memory = eq.memory(oom["S"].to_numpy(float), oom["B"].to_numpy(float)) / 2**20
    ax.scatter(
        oom["S"],
        oom_memory,
        marker="x",
        color="red",
        s=50,
        zorder=4,
        label="OOM (at predicted value)",
    )
    save(fig, out / "memory.png")

    fig, _ = family_figure(
        measured,
        "latency",
        lambda s, b: eq.latency(s, b, theta),
        "latency (ms)",
        "Latency",
        x_axes=("B", "S"),
        scale=1e3,
    )
    save(fig, out / "latency.png")

    fig, _ = family_figure(
        measured.assign(throughput=measured["B"] / measured["latency"]),
        "throughput",
        lambda s, b: b / eq.latency(s, b, theta),
        "throughput (images / s)",
        "Throughput",
        x_axes=("B",),
    )
    save(fig, out / "throughput.png")

    predicted = eq.latency(sizes, batches, theta)
    save(
        parity_figure(measured, "latency", predicted, "latency (s)"),
        out / "latency_parity.png",
    )

    fig, _ = family_figure(
        measured,
        "energy",
        lambda s, b: eq.energy(s, b, theta_energy),
        "energy per forward pass (J)",
        "Energy",
        x_axes=("B", "S"),
    )
    save(fig, out / "energy.png")

    predicted = eq.energy(sizes, batches, theta_energy)
    save(
        parity_figure(measured, "energy", predicted, "energy (J)"), out / "energy_parity.png"
    )


if __name__ == "__main__":
    main()
