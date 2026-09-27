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


def grid_figure(frame, column, predict, x, ylabel, title, scale=1.0):
    """One small panel per value of the other variable: the equation line and its measured points.

    The panel title shows the mean relative error of the equation on that panel's points.
    """
    panel = "B" if x == "S" else "S"
    values = sorted(frame[panel].unique())
    ncols = 4
    nrows = -(-len(values) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows), sharex=True)
    x_curve = np.geomspace(frame[x].min(), frame[x].max(), 200)
    for ax, value in zip(axes.flat, values):
        s, b = (x_curve, value) if x == "S" else (value, x_curve)
        ax.plot(x_curve, predict(s, b) * scale, color="gray")
        points = frame[(frame[panel] == value) & frame[column].notna()]
        draw_points(ax, points[x], points[column] * scale, "royalblue", points["is_validation"])
        s, b = points["S"].to_numpy(float), points["B"].to_numpy(float)
        error = np.mean(np.abs(predict(s, b) * scale / (points[column] * scale) - 1))
        ax.set_title(f"{panel} = {value}   error {error:.0%}", fontsize=9)
        ax.set(yscale="log")
        ax.set_xscale("log", base=2)
    for ax in axes.flat[len(values) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(AXIS_LABELS[x])
        ax.xaxis.set_tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    fig.legend(
        handles=[
            plt.Line2D([], [], color="gray", label="equation"),
            plt.Line2D([], [], marker="o", ls="", color="royalblue", label="measured, calibration"),
            plt.Line2D(
                [],
                [],
                marker="o",
                ls="",
                mfc="white",
                mec="royalblue",
                label="measured, validation",
            ),
        ],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.suptitle(title, y=1.04)
    return fig


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
    fig = grid_figure(
        frame,
        "flops_counted",
        lambda s, b: eq.flops(s, b) - (eq.HEAD_HIDDEN + eq.NUM_CLASSES) * b,
        "S",
        "GFLOPs",
        "FLOPs per forward pass: equation vs torch FlopCounterMode",
        scale=1e-9,
    )
    save(fig, out / "flops.png")

    all_predicted = eq.memory(frame["S"].to_numpy(float), frame["B"].to_numpy(float))
    capacity = meta["gpu_total_memory_bytes"]
    fig = grid_figure(
        frame,
        "memory",
        eq.memory,
        "S",
        "memory (MiB)",
        f"Memory: equation vs max_memory_allocated()   |   OOM measured "
        f"{int(frame['is_oom'].sum())}, predicted {int((all_predicted > capacity).sum())} "
        f"(max predicted {all_predicted.max() / 2**30:.1f} GiB, device {capacity / 2**30:.1f} GiB)",
        scale=2**-20,
    )
    save(fig, out / "memory.png")

    fig = grid_figure(
        measured,
        "latency",
        lambda s, b: eq.latency(s, b, theta),
        "B",
        "latency (ms)",
        "Latency per forward pass",
        scale=1e3,
    )
    save(fig, out / "latency.png")

    fig = grid_figure(
        measured.assign(throughput=measured["B"] / measured["latency"]),
        "throughput",
        lambda s, b: b / eq.latency(s, b, theta),
        "B",
        "images / s",
        "Throughput",
    )
    save(fig, out / "throughput.png")

    predicted = eq.latency(sizes, batches, theta)
    save(
        parity_figure(measured, "latency", predicted, "latency (s)"),
        out / "latency_parity.png",
    )

    fig = grid_figure(
        measured,
        "energy",
        lambda s, b: eq.energy(s, b, theta_energy),
        "B",
        "energy (J)",
        "Energy per forward pass (whole GPU)",
    )
    save(fig, out / "energy.png")

    predicted = eq.energy(sizes, batches, theta_energy)
    save(parity_figure(measured, "energy", predicted, "energy (J)"), out / "energy_parity.png")

if __name__ == "__main__":
    main()
