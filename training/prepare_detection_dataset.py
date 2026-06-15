"""Build a YOLO **detection** dataset from cell-annotated 4x4 challenge images.

Consumes ``annotations.jsonl`` (from ``annotate_detection_cli.py``) where each record is
``{image_path, label, cells:[1..16]}``. Each selected cell becomes one YOLO bounding box
(cell-level weak supervision). Emits the standard YOLO detect layout::

    detection_dataset/
      images/{train,val}/*.png
      labels/{train,val}/*.txt    # "<class_id> <cx> <cy> <w> <h>" per box (normalized)
      data.yaml

Usage::

    python training/prepare_detection_dataset.py \\
        --annotations collected/full/annotations.jsonl --dataset training/detection_dataset
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import class_mapping
import click


def cell_to_yolo_bbox(cell: int, grid: int = 4) -> tuple[float, float, float, float]:
    """Convert a 1-indexed grid cell to a normalized YOLO bbox (cx, cy, w, h).

    Args:
        cell: 1-indexed cell number (1..grid*grid).
        grid: Cells per row/column (4 for 4x4).

    Returns:
        (cx, cy, w, h) normalized to [0, 1], the cell's full square.

    Raises:
        ValueError: If cell is out of range.
    """
    if not 1 <= cell <= grid * grid:
        raise ValueError(f"cell {cell} out of range for grid {grid}x{grid}")
    idx = cell - 1
    row, col = divmod(idx, grid)
    size = 1.0 / grid
    return ((col + 0.5) * size, (row + 0.5) * size, size, size)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def prepare(
    annotations: Path,
    dataset: Path,
    val_split: float = 0.1,
    seed: int = 0,
    grid: int = 4,
) -> dict[str, Any]:
    """Build the YOLO detection dataset from annotations.

    Args:
        annotations: Path to annotations.jsonl.
        dataset: Output dataset root.
        val_split: Validation fraction.
        seed: Deterministic shuffle seed.
        grid: Grid size (4 for 4x4).

    Returns:
        Summary dict (images, train, val, skipped).
    """
    if not 0.0 <= val_split <= 1.0:
        raise ValueError(f"val_split must be between 0.0 and 1.0, got {val_split}")

    records = _load_jsonl(Path(annotations))
    valid: list[tuple[Path, int, list[int]]] = []
    skipped = 0
    for rec in records:
        image_path = rec.get("image_path")
        label = rec.get("label")
        cells = rec.get("cells") or []
        if not image_path or not label or not cells:
            skipped += 1
            continue
        try:
            class_id = class_mapping.detection_class_id(str(label))
        except KeyError:
            skipped += 1
            continue
        valid.append((Path(image_path), class_id, list(cells)))

    rng = random.Random(seed)
    rng.shuffle(valid)
    n_val = round(len(valid) * val_split)
    splits = {"val": valid[:n_val], "train": valid[n_val:]}

    for split, items in splits.items():
        (Path(dataset) / "images" / split).mkdir(parents=True, exist_ok=True)
        (Path(dataset) / "labels" / split).mkdir(parents=True, exist_ok=True)
        for src, class_id, cells in items:
            dest_img = Path(dataset) / "images" / split / src.name
            shutil.copy2(src, dest_img)
            lines = []
            for cell in cells:
                cx, cy, w, h = cell_to_yolo_bbox(cell, grid)
                lines.append(f"{class_id} {cx} {cy} {w} {h}")
            label_file = Path(dataset) / "labels" / split / f"{src.stem}.txt"
            label_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write_data_yaml(Path(dataset))

    return {
        "images": len(valid),
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "skipped": skipped,
    }


def _write_data_yaml(dataset: Path) -> None:
    """Write data.yaml (paths + ordered class names) for ultralytics detect training."""
    names_block = "\n".join(
        f"  {i}: {name}" for i, name in enumerate(class_mapping.DETECTION_CLASSES)
    )
    content = (
        f"path: {dataset.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(class_mapping.DETECTION_CLASSES)}\n"
        "names:\n"
        f"{names_block}\n"
    )
    (dataset / "data.yaml").write_text(content, encoding="utf-8")


@click.command()
@click.option(
    "--annotations",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("collected/full/annotations.jsonl"),
    help="Path to annotations.jsonl from annotate_detection_cli.py.",
)
@click.option(
    "--dataset",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("training/detection_dataset"),
    help="Output YOLO detection dataset root.",
)
@click.option("--val-split", type=float, default=0.1, help="Validation fraction.")
@click.option("--seed", type=int, default=0, help="Deterministic shuffle seed.")
def main(annotations: Path, dataset: Path, val_split: float, seed: int) -> None:
    """CLI: build the YOLO detection dataset."""
    summary = prepare(annotations=annotations, dataset=dataset, val_split=val_split, seed=seed)
    click.echo(
        f"Built detection dataset: {summary['images']} images "
        f"(train={summary['train']}, val={summary['val']}, skipped={summary['skipped']}) "
        f"-> {dataset}"
    )


if __name__ == "__main__":
    main()
