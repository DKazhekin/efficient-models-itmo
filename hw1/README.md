# HW1 — Analytical performance model of a small CNN

## Setup

| | |
|---|---|
| GPU | _fill from `results/meta.json`_ (Colab Tesla T4, 15 GB) |
| Driver / CUDA / cuDNN | _from meta.json_ |
| PyTorch / Python | _from meta.json_ |
| Energy source | NVML cumulative energy counter |

## Reproduce

Open `run_colab.ipynb` on Colab with a T4 runtime, set `REPO_URL`, run all. Or by hand:

```bash
pip install -r requirements.txt
python measure.py      # ~20 min on T4 -> results/measurements.csv, meta.json
python calibrate.py    # -> results/theta.json
python plots.py        # -> results/figures/*.png
pytest tests           # CPU-only checks of the equations
```

Grid: S ∈ {32, 64, 128, 224, 256, 384, 512} × B ∈ {1, 2, 4, …, 256} is used for calibration; 4
extra sizes and 3 extra non-power-of-two batches (seed 2026) are validation-only. Configs run in
random order after a 20 s GPU warm-up, so thermal drift is not correlated with (S, B).

## Equations

Derivation: `derivation.md` / `hw1_handwritten.pdf`.

| Function | Formula |
|----------|---------|
| FLOPs | 17 712 · B · S² + 313 700 · B |
| Memory | 4.18 MB + 188 · B · S² + 3 472 · B bytes (all activations at once, course convention) |
| Latency | max(t_launch, Σ_ops max(F_op / π, Q_op / β)) |
| Energy | P_static · T + P_dynamic · T_gpu |

## Results

_Fill from `results/theta.json` after the run._

| | θ | calibration MAPE | validation MAPE |
|---|---|---|---|
| FLOPs | – | – (vs FlopCounterMode: exact) | |
| Memory | – | | |
| Latency | t_launch, π (β from copy benchmark) | | |
| Energy | P_static, P_dynamic | | |

## Discussion

_To be written from the measured data._
