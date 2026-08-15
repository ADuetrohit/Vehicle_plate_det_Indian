"""Crop loading, double-row unfolding, and augmentation for plate OCR."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from charset import encode
from model import IMAGE_HEIGHT, IMAGE_WIDTH, LAYOUT_SIZE

LAYOUT_TO_INDEX = {"single": 0, "double": 1}


def _row_ink_profile(image: np.ndarray) -> np.ndarray:
    """Per-row glyph energy, used to locate text bands.

    Horizontal gradient responds to the vertical strokes that make up digits and
    letters, so text rows score high while the plate face, the gap between rows,
    and the padding score low. Gradient is used rather than raw intensity so the
    profile does not depend on plate colour -- this corpus mixes white, yellow
    and green faces with both dark and light glyphs.
    """
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    gradient = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3))
    profile = gradient.sum(axis=1)
    # Smooth over a few rows so a single noisy scanline can not create a split.
    kernel = np.ones(3, dtype=np.float32) / 3.0
    return np.convolve(profile, kernel, mode="same")


def unfold_double_row(image: np.ndarray, overlap: float = 0.06) -> np.ndarray:
    """Flatten a two-line plate into one left-to-right strip.

    The bottom row is placed to the right of the top row, which is the reading
    order of an Indian double-row plate, so a CTC head can read it as one
    sequence.

    The cut can not simply halve the crop. These crops are aspect-preserving
    padded squares, so the plate occupies a band somewhere inside the frame and
    a central cut would slice through padding while leaving both text rows in
    both halves. Instead the text band is located from the ink profile and the
    split is placed at the quietest row inside it -- the gap between the two
    lines. A small overlap is kept on each half so ascenders and descenders
    survive the cut.

    Falls back to a central split when no clear two-band structure is found,
    which keeps the function total for crops the layout classifier mislabels.
    """
    height, width = image.shape[:2]
    if height < 8:
        return image

    profile = _row_ink_profile(image)
    peak = float(profile.max())
    if peak <= 0:
        return image

    # Vertical extent of anything that looks like text.
    active = np.flatnonzero(profile > 0.25 * peak)
    if active.size < 4:
        top_y, bottom_y = 0, height
    else:
        top_y, bottom_y = int(active[0]), int(active[-1]) + 1

    band = bottom_y - top_y
    if band < 8:
        top_y, bottom_y, band = 0, height, height

    # Look for the quietest row in the middle half of the text band: on a
    # double-row plate that is the inter-row gap.
    search_lo = top_y + band // 4
    search_hi = bottom_y - band // 4
    if search_hi - search_lo >= 2:
        split = search_lo + int(np.argmin(profile[search_lo:search_hi]))
    else:
        split = top_y + band // 2

    margin = max(1, int(round(band * overlap)))
    top = image[top_y : min(split + margin, height)]
    bottom = image[max(split - margin, 0) : bottom_y]
    if top.size == 0 or bottom.size == 0:
        return image

    target_h = max(top.shape[0], bottom.shape[0])
    top = cv2.resize(top, (width, target_h), interpolation=cv2.INTER_LINEAR)
    bottom = cv2.resize(bottom, (width, target_h), interpolation=cv2.INTER_LINEAR)
    return np.hstack([top, bottom])


def resize_keep_ratio(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize into a fixed box, preserving aspect and padding with edge pixels.

    Stretching a plate to a fixed box distorts glyph width, and glyph width is
    one of the few cues separating ``0`` from ``O`` and ``8`` from ``B``, so the
    aspect ratio is preserved and the remainder padded instead.
    """
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((height, width), int(np.median(resized)), dtype=resized.dtype)
    y0 = (height - new_h) // 2
    x0 = (width - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _augment(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Photometric and mild geometric jitter.

    The synthetic corpus already carries night, rain, fog, glare, motion and
    compression effects, so augmentation here only needs to cover the residual
    gap to real camera output: sensor noise, focus softness, and the small
    rotation a pole-mounted camera introduces.
    """
    if rng.random() < 0.5:
        alpha = rng.uniform(0.7, 1.3)
        beta = rng.uniform(-30, 30)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        image = cv2.GaussianBlur(image, (k, k), 0)

    if rng.random() < 0.3:
        noise = np.random.normal(0, rng.uniform(3, 12), image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.4:
        angle = rng.uniform(-4, 4)
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if rng.random() < 0.25:
        quality = rng.randint(30, 80)
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            image = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return image


class PlateOCRDataset(Dataset):
    """Crops paired with registration text, for CTC training.

    ``unfold`` controls double-row handling. During training the ground-truth
    layout from the generation manifest is used. At inference the layout is not
    known, so :class:`model.LayoutClassifier` supplies it -- see
    :mod:`evaluate`.
    """

    def __init__(
        self,
        csv_path: Path,
        root: Path,
        augment: bool = False,
        unfold: bool = True,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.augment = augment
        self.unfold = unfold
        self.rng = random.Random(seed)
        with Path(csv_path).open(encoding="utf-8", newline="") as handle:
            self.records = list(csv.DictReader(handle))
        if not self.records:
            raise ValueError(f"no rows in {csv_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        path = self.root / record["image_path"]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"unreadable crop: {path}")

        if self.unfold and record.get("layout") == "double":
            image = unfold_double_row(image)

        if self.augment:
            image = _augment(image, self.rng)

        image = resize_keep_ratio(image, IMAGE_HEIGHT, IMAGE_WIDTH)
        tensor = torch.from_numpy(image).float().div_(255.0).sub_(0.5).div_(0.5)
        tensor = tensor.unsqueeze(0)

        text = record["plate_text"]
        targets = torch.tensor(encode(text), dtype=torch.long)
        return tensor, targets, len(targets), text


class LayoutDataset(Dataset):
    """Crops paired with the single/double layout label.

    Only synthetic rows carry a known layout, so rows labelled ``unknown`` are
    dropped rather than assumed single.
    """

    def __init__(self, csv_path: Path, root: Path, augment: bool = False, seed: int = 0) -> None:
        self.root = Path(root)
        self.augment = augment
        self.rng = random.Random(seed)
        with Path(csv_path).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.records = [r for r in rows if r.get("layout") in LAYOUT_TO_INDEX]
        if not self.records:
            raise ValueError(f"no rows with a known layout in {csv_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = cv2.imread(str(self.root / record["image_path"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(record["image_path"])
        if self.augment:
            image = _augment(image, self.rng)
        image = resize_keep_ratio(image, LAYOUT_SIZE, LAYOUT_SIZE)
        tensor = torch.from_numpy(image).float().div_(255.0).sub_(0.5).div_(0.5).unsqueeze(0)
        return tensor, LAYOUT_TO_INDEX[record["layout"]]


def ctc_collate(batch):
    """Stack images and concatenate targets in the flat form ``CTCLoss`` wants."""
    images, targets, lengths, texts = zip(*batch)
    return (
        torch.stack(images, 0),
        torch.cat(targets, 0),
        torch.tensor(lengths, dtype=torch.long),
        list(texts),
    )
