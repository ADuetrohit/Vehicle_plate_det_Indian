"""Models for plate OCR: a layout classifier and a CNN+CTC reader.

Two small networks rather than one, because the corpus is 20% double-row and a
CTC head can not read stacked text. CTC assumes the sequence runs left to right,
but on a two-line plate both rows occupy the same columns, so their characters
interleave in the feature sequence and the alignment is unrecoverable.

The detector box can not separate the two layouts either: measured on this
dataset the median box aspect is 3.94 for single-row and 4.04 for double-row,
because perspective warp dominates the axis-aligned box. So layout is predicted
from the crop itself, and double-row crops are unfolded into one wide strip
before reading -- see :func:`dataset.unfold_double_row`.

Both networks target a Raspberry Pi 3B+ CPU: modest channel widths, no
recurrent layer by default, and a fully convolutional path that exports cleanly
to ONNX.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from charset import NUM_CLASSES

#: Reader input geometry. Width drives the CTC time-step count.
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 128

#: Layout classifier input geometry.
LAYOUT_SIZE = 64


class LayoutClassifier(nn.Module):
    """Single-row vs double-row from a 64x64 grayscale crop.

    Deliberately tiny: the cue is whether glyph mass forms one horizontal band
    or two, a low-frequency global property, so a few strided convolutions and a
    global average pool suffice.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1))


def _conv_bn(in_ch: int, out_ch: int, stride=1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class PlateReader(nn.Module):
    """CNN + CTC plate reader.

    Height is collapsed to 1 while width is preserved, turning the feature map
    into a left-to-right sequence. With a 32x128 input the shapes run:

    ==============================  ===============
    stage                           output (C,H,W)
    ==============================  ===============
    input                            (1, 32, 128)
    conv 32  + pool 2x2              (32, 16, 64)
    conv 64  + pool 2x2              (64, 8, 32)
    conv 128 x2 + pool (2,1)         (128, 4, 32)
    conv 256 + pool (2,1)            (256, 2, 32)
    conv 256, kernel (2,1), valid    (256, 1, 32)
    ==============================  ===============

    That yields ``T = 32`` time steps for at most 11 characters, leaving CTC
    ample room for blanks between repeated glyphs.

    Set ``use_lstm=True`` to add a bidirectional LSTM over the sequence. It
    raises accuracy on ambiguous glyphs but costs roughly 3x the CPU time, so it
    is off by default for the Pi target.
    """

    def __init__(self, use_lstm: bool = False, dropout: float = 0.25) -> None:
        super().__init__()
        self.use_lstm = use_lstm
        self.backbone = nn.Sequential(
            _conv_bn(1, 32),
            nn.MaxPool2d(2, 2),
            _conv_bn(32, 64),
            nn.MaxPool2d(2, 2),
            _conv_bn(64, 128),
            _conv_bn(128, 128),
            nn.MaxPool2d((2, 1), (2, 1)),
            _conv_bn(128, 256),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Dropout2d(dropout),
            nn.Conv2d(256, 256, kernel_size=(2, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        if use_lstm:
            self.rnn = nn.LSTM(256, 128, num_layers=2, bidirectional=True, batch_first=False)
            self.classifier = nn.Linear(256, NUM_CLASSES)
        else:
            self.rnn = None
            self.classifier = nn.Linear(256, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(T, N, NUM_CLASSES)`` log-probabilities for ``CTCLoss``."""
        features = self.backbone(x)
        if features.size(2) != 1:
            raise RuntimeError(
                f"expected height 1 after backbone, got {features.size(2)}; "
                f"input must be {IMAGE_HEIGHT}x{IMAGE_WIDTH}"
            )
        sequence = features.squeeze(2).permute(2, 0, 1)  # (T, N, C)
        if self.rnn is not None:
            sequence, _ = self.rnn(sequence)
        return self.classifier(sequence).log_softmax(2)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
