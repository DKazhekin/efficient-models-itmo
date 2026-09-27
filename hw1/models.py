from torch import nn

from equations import CONV_LAYERS, HEAD_HIDDEN, NUM_CLASSES


def build_model():
    """Sequential CNN, 3 x S x S image -> 100 logits."""
    layers = []
    for index, (c_in, c_out, kernel, stride) in enumerate(CONV_LAYERS):
        layers += [
            nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        ]
        if index == 0:
            layers.append(nn.MaxPool2d(3, stride=2, padding=1))
    layers += [
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(CONV_LAYERS[-1][1], HEAD_HIDDEN),
        nn.ReLU(inplace=True),
        nn.Linear(HEAD_HIDDEN, NUM_CLASSES),
    ]
    return nn.Sequential(*layers)
