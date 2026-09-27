import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import equations as eq  # noqa: E402
from models import build_model  # noqa: E402

THETA = {"launch_floor": 1e-3, "peak_flops": 4e12, "bandwidth": 2.5e11}


@pytest.mark.parametrize(("image_size", "batch"), [(32, 1), (64, 3), (224, 2), (512, 1)])
def test_flops_match_pytorch_flop_counter(image_size, batch):
    inputs = torch.randn(batch, 3, image_size, image_size)
    with torch.inference_mode(), FlopCounterMode(display=False) as counter:
        build_model().eval()(inputs)

    # FlopCounterMode does not count the bias adds of the linear layers
    bias_adds = batch * (eq.HEAD_HIDDEN + eq.NUM_CLASSES)
    assert eq.flops(image_size, batch) == pytest.approx(
        counter.get_total_flops() + bias_adds, rel=1e-12
    )


@pytest.mark.parametrize(("image_size", "batch"), [(32, 1), (100, 7), (512, 256)])
def test_flops_closed_form(image_size, batch):
    # hand-derived coefficients, see derivation.md
    assert eq.flops(image_size, batch) == 17_712 * batch * image_size**2 + 313_700 * batch


def test_parameter_bytes_match_model_state():
    model = build_model()
    tensors = list(model.parameters()) + list(model.buffers())
    assert sorted(eq.parameter_bytes()) == sorted(t.numel() * t.element_size() for t in tensors)


@pytest.mark.parametrize(("image_size", "batch"), [(32, 1), (256, 8)])
def test_memory_closed_form(image_size, batch):
    # input 12 + all activations 176 bytes per pixel; head outputs 4 * (512 + 256 + 100) per image
    pixels = batch * image_size**2
    expected = sum(eq.parameter_bytes()) + 188 * pixels + 3_472 * batch
    assert eq.memory(image_size, batch) == pytest.approx(expected, rel=1e-12)


def test_functions_broadcast_over_grid():
    sizes, batches = np.meshgrid([32, 64, 128], [1, 2, 4, 8])
    theta_energy = {"static_power": 30.0, "dynamic_power": 40.0, "latency": THETA}
    for value in (
        eq.flops(sizes, batches),
        eq.memory(sizes, batches),
        eq.latency(sizes, batches, THETA),
        eq.energy(sizes, batches, theta_energy),
    ):
        assert value.shape == sizes.shape
        assert np.all(value > 0)
