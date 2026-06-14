"""Write a model_card.json sidecar for a published model (classification or detection).

Records provenance for a shipped ONNX: date, task, classes, sha256, dataset size, hyperparams.
Pair with ``compute_sha256.py`` so the card's sha256 matches what the solver verifies.

Usage::

    python training/write_model_card.py --onnx best.onnx --task detect \\
        --epochs 100 --imgsz 640 --dataset-size 1200 --out best.model_card.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import class_mapping
import click
from compute_sha256 import compute_sha256


def build_card(
    onnx: Path,
    task: str,
    epochs: int,
    imgsz: int,
    dataset_size: int,
    date: str,
) -> dict[str, Any]:
    """Build the model_card dict (does not write).

    Args:
        onnx: Path to the exported ONNX (hashed for the card).
        task: "classify" or "detect".
        epochs: Training epochs.
        imgsz: Input image size.
        dataset_size: Number of training samples.
        date: ISO date string (caller-supplied; scripts avoid wall-clock for determinism).

    Returns:
        The model_card dict.
    """
    classes = (
        class_mapping.DETECTION_CLASSES
        if task == "detect"
        else [class_mapping.FOLDER_TO_LABEL[f] for f in class_mapping.FOLDER_ORDER]
    )
    return {
        "date": date,
        "task": task,
        "classes": classes,
        "num_classes": len(classes),
        "epochs": epochs,
        "imgsz": imgsz,
        "dataset_size": dataset_size,
        "sha256": compute_sha256(onnx),
    }


@click.command()
@click.option(
    "--onnx", type=click.Path(path_type=Path), required=True, help="Exported ONNX to hash."
)
@click.option(
    "--task", type=click.Choice(["classify", "detect"]), required=True, help="Model task."
)
@click.option("--epochs", type=int, default=0, help="Training epochs.")
@click.option("--imgsz", type=int, default=640, help="Input image size.")
@click.option("--dataset-size", type=int, default=0, help="Number of training samples.")
@click.option("--date", default="", help="ISO date (e.g. 2026-06-14).")
@click.option("--out", type=click.Path(path_type=Path), default=None, help="Output JSON path.")
def main(
    onnx: Path, task: str, epochs: int, imgsz: int, dataset_size: int, date: str, out: Path | None
) -> None:
    """CLI: write a model_card.json next to the ONNX (or to --out)."""
    card = build_card(onnx, task, epochs, imgsz, dataset_size, date)
    out_path = out or onnx.with_suffix(".model_card.json")
    out_path.write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    click.echo(f"Wrote model card -> {out_path}")
    click.echo(json.dumps(card, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
