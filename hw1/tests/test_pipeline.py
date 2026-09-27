import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW_DIR))

import equations as eq  # noqa: E402
from measure import BASE_BATCHES, BASE_SIZES, sample_grid  # noqa: E402

TRUE_THETA = {"launch_floor": 1.2e-3, "peak_flops": 4.0e12, "bandwidth": 2.2e11}
TRUE_ENERGY = {"static_power": 30.0, "dynamic_power": 35.0}
NOISE = 0.05
T4_MEMORY_BYTES = 15_828_320_256


def _synthetic_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    random_sizes, random_batches = sample_grid(2026)
    rows = []
    for size in sorted(set(BASE_SIZES) | set(random_sizes)):
        for batch in sorted(set(BASE_BATCHES) | set(random_batches)):
            theta_energy = TRUE_ENERGY | {"latency": TRUE_THETA}
            rows.append(
                {
                    "S": size,
                    "B": batch,
                    "latency": eq.latency(size, batch, TRUE_THETA) * rng.lognormal(0, NOISE),
                    "memory": eq.memory(size, batch),
                    "energy": eq.energy(size, batch, theta_energy) * rng.lognormal(0, NOISE),
                    "is_validation": size in random_sizes or batch in random_batches,
                    "status": "ok",
                    "flops_counted": eq.flops(size, batch)
                    - (eq.HEAD_HIDDEN + eq.NUM_CLASSES) * batch,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def results_dir(tmp_path: Path) -> Path:
    _synthetic_frame().to_csv(tmp_path / "measurements.csv", index=False)
    meta = {
        "gpu_total_memory_bytes": T4_MEMORY_BYTES,
        "copy_bandwidth": TRUE_THETA["bandwidth"],
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return tmp_path


def test_random_grid_avoids_base_grid_and_powers_of_two():
    sizes, batches = sample_grid(2026)
    assert len(sizes) == 4 and len(batches) == 3
    assert all(s % 16 == 0 and s not in BASE_SIZES for s in sizes)
    assert all(b & (b - 1) != 0 for b in batches)


def test_calibration_recovers_theta_and_plots_render(results_dir: Path):
    for script in ("calibrate.py", "plots.py"):
        subprocess.run(
            [sys.executable, script, "--results", str(results_dir)], cwd=HW_DIR, check=True
        )

    fitted = json.loads((results_dir / "theta.json").read_text())
    for key, value in TRUE_THETA.items():
        assert fitted["latency"][key] == pytest.approx(value, rel=0.1)
    for key, value in TRUE_ENERGY.items():
        assert fitted["energy"][key] == pytest.approx(value, rel=0.15)
    assert fitted["metrics"]["latency"]["validation"]["mape"] < 0.1
    assert fitted["metrics"]["oom"]["predicted_over_total_memory"] == 0
    assert len(list((results_dir / "figures").glob("*.png"))) == 7
