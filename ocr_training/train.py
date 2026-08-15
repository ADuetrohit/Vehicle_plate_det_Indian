"""Train the plate reader (CTC) and the layout classifier.

Usage::

    python ocr_training/train.py --task reader --epochs 40
    python ocr_training/train.py --task layout --epochs 6

Both tasks read the manifests written by ``prepare_ocr_dataset.py``. Training
and validation are synthetic; the test split is real photography and is never
touched here -- run ``evaluate.py`` for that, once.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from charset import BLANK_INDEX, ctc_greedy_decode
from dataset import LayoutDataset, PlateOCRDataset, ctc_collate
from model import LayoutClassifier, PlateReader, count_parameters


def _device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate_reader(model, loader, device) -> dict:
    """Exact-match and character-level accuracy over a loader."""
    model.eval()
    exact = total = 0
    char_hits = char_total = 0
    for images, _, _, texts in loader:
        images = images.to(device)
        log_probs = model(images)
        predictions = log_probs.argmax(2).permute(1, 0).cpu().tolist()
        for path, truth in zip(predictions, texts):
            guess = ctc_greedy_decode(path)
            total += 1
            exact += int(guess == truth)
            char_total += len(truth)
            char_hits += sum(1 for a, b in zip(guess, truth) if a == b)
    return {
        "exact_match": exact / max(total, 1),
        "char_accuracy": char_hits / max(char_total, 1),
        "samples": total,
    }


def train_reader(args, root: Path, splits: Path, device) -> dict:
    train_set = PlateOCRDataset(splits / "train.csv", root, augment=True, seed=args.seed)
    val_set = PlateOCRDataset(splits / "val.csv", root, augment=False)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=ctc_collate, drop_last=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=ctc_collate,
    )

    model = PlateReader(use_lstm=args.lstm).to(device)
    print(f"reader parameters: {count_parameters(model):,}")

    # zero_infinity guards the case where a target is longer than the input
    # sequence; such a batch would otherwise poison the run with inf loss.
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader)
    )

    best = 0.0
    history = []
    out_dir = root / "ocr_training" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        started = time.time()
        for step, (images, targets, lengths, _) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device)
            log_probs = model(images)
            input_lengths = torch.full(
                (images.size(0),), log_probs.size(0), dtype=torch.long, device=device
            )
            loss = criterion(log_probs, targets, input_lengths, lengths.to(device))

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            scheduler.step()
            running += loss.item()
            if step % args.log_every == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {running/step:.4f}")

        metrics = evaluate_reader(model, val_loader, device)
        metrics.update(epoch=epoch, loss=running / len(train_loader), seconds=time.time() - started)
        history.append(metrics)
        print(
            f"epoch {epoch}: loss {metrics['loss']:.4f} "
            f"val exact {metrics['exact_match']:.4f} char {metrics['char_accuracy']:.4f} "
            f"({metrics['seconds']:.0f}s)"
        )

        if metrics["exact_match"] >= best:
            best = metrics["exact_match"]
            torch.save(
                {"model": model.state_dict(), "use_lstm": args.lstm, "metrics": metrics},
                out_dir / "reader_best.pt",
            )
            print(f"  saved reader_best.pt (val exact {best:.4f})")

    (out_dir / "reader_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"best_val_exact_match": best, "epochs": args.epochs}


def train_layout(args, root: Path, splits: Path, device) -> dict:
    train_set = LayoutDataset(splits / "train.csv", root, augment=True, seed=args.seed)
    val_set = LayoutDataset(splits / "val.csv", root, augment=False)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        drop_last=True, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers)

    model = LayoutClassifier().to(device)
    print(f"layout parameters: {count_parameters(model):,}")

    # Roughly 80/20 single/double, so the loss is weighted to stop the model
    # collapsing onto the majority layout.
    counts = [0, 0]
    for record in train_set.records:
        counts[0 if record["layout"] == "single" else 1] += 1
    weights = torch.tensor(
        [len(train_set) / (2 * max(c, 1)) for c in counts], dtype=torch.float, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = root / "ocr_training" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(model(images), labels)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            running += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                correct += (model(images).argmax(1) == labels).sum().item()
                total += labels.numel()
        accuracy = correct / max(total, 1)
        print(f"epoch {epoch}: loss {running/len(train_loader):.4f} val accuracy {accuracy:.4f}")
        if accuracy >= best:
            best = accuracy
            torch.save({"model": model.state_dict(), "accuracy": accuracy}, out_dir / "layout_best.pt")
    return {"best_val_accuracy": best}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["reader", "layout"], default="reader")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--lstm", action="store_true", help="add a BiLSTM to the reader")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--limit-batches", type=int, default=0, help="smoke-test only")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = _device(args.device)
    print(f"device: {device}")

    splits = args.root / "ocr_training" / "splits"
    if not (splits / "train.csv").exists():
        raise SystemExit("run prepare_ocr_dataset.py first")

    result = (train_reader if args.task == "reader" else train_layout)(args, args.root, splits, device)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
