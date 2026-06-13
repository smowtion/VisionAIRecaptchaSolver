"""Merge human-reviewed tiles into a YOLO-classification ImageFolder dataset.

Reads ``reviewed.jsonl`` (produced by ``review_cli.py``) and copies each kept tile into
``<dataset>/<train|val>/<CanonicalFolder>/``. Labels are normalized via ``class_mapping``
so the dataset folders always match the model's class taxonomy. The train/val split is
stratified per class and deterministic for a given seed.

Usage::

    python training/prepare_dataset.py --reviewed collected/reviewed.jsonl \\
        --dataset training/dataset --val-split 0.1
"""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import class_mapping
import click

_SKIP_ACTIONS = {"discard", "skip"}


def _load_reviewed(reviewed: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in reviewed.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def merge(
    reviewed: Path,
    dataset: Path,
    val_split: float = 0.1,
    seed: int = 0,
) -> dict[str, Any]:
    """Copy reviewed/kept tiles into an ImageFolder dataset, split train/val.

    Args:
        reviewed: Path to reviewed.jsonl.
        dataset: Output dataset root (``<dataset>/train`` and ``<dataset>/val`` created).
        val_split: Fraction of each class routed to validation (0.0-1.0).
        seed: RNG seed for the deterministic shuffle.

    Returns:
        Summary dict with copied/skipped/train/val counts and a per-class breakdown.
    """
    if not 0.0 <= val_split <= 1.0:
        raise ValueError(f"val_split must be between 0.0 and 1.0, got {val_split}")

    records = _load_reviewed(Path(reviewed))

    by_class: dict[str, list[Path]] = defaultdict(list)
    skipped = 0
    for record in records:
        action = str(record.get("action", "keep")).lower()
        if action in _SKIP_ACTIONS:
            skipped += 1
            continue
        label = record.get("label")
        image_path = record.get("image_path")
        if not label or not image_path:
            skipped += 1
            continue
        folder = class_mapping.normalize_folder(str(label))
        by_class[folder].append(Path(image_path))

    rng = random.Random(seed)
    copied = 0
    train_count = 0
    val_count = 0
    class_summary: dict[str, dict[str, int]] = {}

    for folder in sorted(by_class):
        images = sorted(by_class[folder])
        rng.shuffle(images)
        n_val = round(len(images) * val_split)
        val_images = images[:n_val]
        train_images = images[n_val:]

        for split, split_images in (("train", train_images), ("val", val_images)):
            dest_dir = Path(dataset) / split / folder
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in split_images:
                shutil.copy2(src, dest_dir / src.name)

        copied += len(images)
        train_count += len(train_images)
        val_count += len(val_images)
        class_summary[folder] = {"train": len(train_images), "val": len(val_images)}

    return {
        "copied": copied,
        "skipped": skipped,
        "train": train_count,
        "val": val_count,
        "by_class": class_summary,
    }


@click.command()
@click.option(
    "--reviewed",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("collected/reviewed.jsonl"),
    help="Path to reviewed.jsonl produced by review_cli.py.",
)
@click.option(
    "--dataset",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("training/dataset"),
    help="Output ImageFolder dataset root.",
)
@click.option("--val-split", type=float, default=0.1, help="Validation fraction per class.")
@click.option("--seed", type=int, default=0, help="Deterministic shuffle seed.")
def main(reviewed: Path, dataset: Path, val_split: float, seed: int) -> None:
    """CLI entry point for merging reviewed tiles into the dataset."""
    summary = merge(reviewed=reviewed, dataset=dataset, val_split=val_split, seed=seed)
    click.echo(
        f"Merged {summary['copied']} tiles "
        f"(train={summary['train']}, val={summary['val']}, skipped={summary['skipped']}) "
        f"into {dataset}"
    )
    for folder, counts in summary["by_class"].items():
        click.echo(f"  {folder}: train={counts['train']} val={counts['val']}")


if __name__ == "__main__":
    main()
