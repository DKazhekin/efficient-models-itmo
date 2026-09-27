import numpy as np

CONV_LAYERS = [
    (3, 32, 7, 2),
    (32, 64, 5, 1),
    (64, 128, 3, 2),
    (128, 256, 1, 1),
    (256, 256, 3, 2),
    (256, 512, 1, 1),
]
HEAD_HIDDEN = 256
NUM_CLASSES = 100


def ops(image_size, batch):
    """Every kernel of the forward pass: FLOPs, bytes moved, bytes allocated."""
    batch = np.asarray(batch, dtype=float)
    pixels = (
        batch * np.asarray(image_size, dtype=float) ** 2
    )  # total numbers of pixels in the batch for the single channel

    result = []
    divisor = 1  # the current feature map is (S / divisor) x (S / divisor)
    for index, (c_in, c_out, kernel, stride) in enumerate(CONV_LAYERS):
        x_in = (
            c_in / divisor**2 * pixels
        )  # total input number of pixels in the batch for the all channels
        divisor *= stride
        x_out = (
            c_out / divisor**2 * pixels
        )  # total output number of pixels in the batch for the all channels (stride is applied, therefore we have magnitude of two times smaller feature map)

        weights = c_out * c_in * kernel**2
        conv = dict(flops=2 * c_in * kernel**2 * x_out, bytes=4 * (x_in + x_out + weights))
        # eval-mode BN is y = x * scale + shift; it reads weight, bias, running mean and var
        bn = dict(flops=2 * x_out, bytes=8 * x_out + 16 * c_out)
        relu = dict(flops=x_out, bytes=8 * x_out)
        result += [
            dict(conv, dense=True, output=4 * x_out),
            dict(bn, dense=False, output=4 * x_out),
            dict(relu, dense=False, output=None),
        ]

        if index == 0:
            x_in = x_out
            divisor *= 2
            x_out = c_out / divisor**2 * pixels
            indices = 8 * x_out  # on CUDA max_pool2d also allocates int64 indices
            result.append(
                dict(
                    dense=False,
                    flops=8 * x_out,
                    bytes=4 * (x_in + x_out) + indices,
                    output=4 * x_out,
                )
            )

    channels = CONV_LAYERS[-1][1]
    x_in = channels / divisor**2 * pixels
    avg_pool = dict(flops=x_in, bytes=4 * x_in + 4 * channels * batch)
    # linear n_in -> n_out: flops = (2 * n_in * n_out + n_out) * B (MACs + bias adds),
    # bytes = input + output per image + weights and bias once per batch
    fc1 = dict(
        flops=(2 * channels * HEAD_HIDDEN + HEAD_HIDDEN) * batch,
        bytes=4 * (channels + HEAD_HIDDEN) * batch + 4 * (channels * HEAD_HIDDEN + HEAD_HIDDEN),
    )
    relu = dict(flops=HEAD_HIDDEN * batch, bytes=8 * HEAD_HIDDEN * batch)
    fc2 = dict(
        flops=(2 * HEAD_HIDDEN * NUM_CLASSES + NUM_CLASSES) * batch,
        bytes=4 * (HEAD_HIDDEN + NUM_CLASSES) * batch
        + 4 * (HEAD_HIDDEN * NUM_CLASSES + NUM_CLASSES),
    )
    result += [
        dict(avg_pool, dense=False, output=4 * channels * batch),
        dict(fc1, dense=True, output=4 * HEAD_HIDDEN * batch),
        dict(relu, dense=False, output=None),
        dict(fc2, dense=True, output=4 * NUM_CLASSES * batch),
    ]

    return result


def parameter_bytes():
    """Size of every parameter and buffer tensor of the model."""
    sizes = []
    for c_in, c_out, kernel, _ in CONV_LAYERS:
        # conv weight; BN weight, bias, running mean, running var, int64 num_batches_tracked
        sizes += [4 * c_out * c_in * kernel**2] + [4 * c_out] * 4 + [8]
    for n_in, n_out in [(CONV_LAYERS[-1][1], HEAD_HIDDEN), (HEAD_HIDDEN, NUM_CLASSES)]:
        sizes += [4 * n_in * n_out, 4 * n_out]
    return sizes


def gpu_time(image_size, batch, theta):
    """Roofline: each op takes max(FLOPs / peak_flops, bytes / bandwidth)."""
    return sum(
        np.maximum(op["flops"] / theta["peak_flops"], op["bytes"] / theta["bandwidth"])
        for op in ops(image_size, batch)
    )


def flops(image_size, batch):
    """FLOPs of the conv and linear layers."""
    return sum(op["flops"] for op in ops(image_size, batch) if op["dense"])


def memory(image_size, batch):
    """Weights + input + all activations of all layers at once (nothing is freed)."""
    input_bytes = 12 * np.asarray(batch, dtype=float) * np.asarray(image_size, dtype=float) ** 2
    activations = sum(op["output"] for op in ops(image_size, batch) if op["output"] is not None)
    return sum(parameter_bytes()) + input_bytes + activations


def latency(image_size, batch, theta):
    """Seconds per forward pass; theta = {"launch_floor", "peak_flops", "bandwidth"}."""
    return np.maximum(theta["launch_floor"], gpu_time(image_size, batch, theta))


def energy(image_size, batch, theta_energy):
    """Joules per forward pass; theta_energy = {"static_power", "dynamic_power", "latency"}."""
    theta = theta_energy["latency"]
    static = theta_energy["static_power"] * latency(image_size, batch, theta)
    dynamic = theta_energy["dynamic_power"] * gpu_time(image_size, batch, theta)
    return static + dynamic
