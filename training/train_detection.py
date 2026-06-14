"""Train the custom 4x4 reCAPTCHA **detection** model (Tier B) on the bbox dataset.

Trains YOLO detection (not classification) on the dataset built by
``prepare_detection_dataset.py`` for the classes the COCO model lacks. Device is
auto-detected (CUDA > MPS > CPU) so it runs on a cloud GPU or an Apple Silicon Mac.
The heavy ``ultralytics`` import is deferred so the module imports / ``--help`` without a GPU.

Usage::

    python training/train_detection.py --data training/detection_dataset/data.yaml      # auto
    python training/train_detection.py --data .../data.yaml --device mps --no-amp        # Apple Silicon
    python training/train_detection.py --resume
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
from device_utils import resolve_device

DEFAULT_BASE_MODEL = "yolo11x.pt"
DEFAULT_PROJECT = "runs/detect"
DEFAULT_NAME = "rec_detect_model"


def train(
    data: str = "training/detection_dataset/data.yaml",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: int | str | None = None,
    workers: int = 8,
    patience: int = 20,
    base_model: str = DEFAULT_BASE_MODEL,
    project: str = DEFAULT_PROJECT,
    name: str = DEFAULT_NAME,
    resume: bool = False,
    amp: bool = True,
) -> Any:
    """Train (or resume) the custom detection model. Requires ultralytics at runtime.

    Args:
        data: Path to the YOLO detect ``data.yaml``.
        epochs: Training epochs.
        imgsz: Input image size.
        batch: Batch size.
        device: CUDA index / "mps" / "cpu". None or "auto" auto-detects (CUDA > MPS > CPU).
        workers: Dataloader workers.
        patience: Early-stopping patience.
        base_model: Pretrained detection base weights to fine-tune from.
        project: Ultralytics project (run output) directory.
        name: Run name under the project directory.
        resume: Resume from ``<project>/<name>/weights/best.pt``.
        amp: Mixed precision. Set False on MPS if it misbehaves.

    Returns:
        The ultralytics training results object.
    """
    from ultralytics import YOLO

    if resume:
        last_weights = Path(project) / name / "weights" / "best.pt"
        if not last_weights.exists():
            raise FileNotFoundError(f"Cannot resume: weights not found at {last_weights}")
        model = YOLO(str(last_weights))
        return model.train(resume=True)

    model = YOLO(base_model)
    return model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=resolve_device(device),
        workers=workers,
        patience=patience,
        project=project,
        name=name,
        amp=amp,
        # No cache=True here (unlike train.py): 4x4 detection images are larger than
        # classification tiles, so RAM caching risks blowing up memory.
    )


@click.command()
@click.option(
    "--data", default="training/detection_dataset/data.yaml", help="YOLO detect data.yaml."
)
@click.option("--epochs", default=100, type=int, help="Training epochs.")
@click.option("--imgsz", default=640, type=int, help="Input image size.")
@click.option("--batch", default=16, type=int, help="Batch size.")
@click.option("--device", default="auto", help="CUDA index / 'mps' / 'cpu' / 'auto' (default).")
@click.option("--workers", default=8, type=int, help="Dataloader workers.")
@click.option("--patience", default=20, type=int, help="Early-stopping patience.")
@click.option("--base-model", default=DEFAULT_BASE_MODEL, help="Pretrained detect base weights.")
@click.option("--project", default=DEFAULT_PROJECT, help="Run output directory.")
@click.option("--name", default=DEFAULT_NAME, help="Run name.")
@click.option("--resume", is_flag=True, help="Resume from last best.pt.")
@click.option("--amp/--no-amp", default=True, help="Mixed precision (use --no-amp on flaky MPS).")
def main(
    data: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    patience: int,
    base_model: str,
    project: str,
    name: str,
    resume: bool,
    amp: bool,
) -> None:
    """CLI entry point: train or resume the detection model (auto-detects CUDA/MPS/CPU)."""
    train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        patience=patience,
        base_model=base_model,
        project=project,
        name=name,
        resume=resume,
        amp=amp,
    )


if __name__ == "__main__":
    main()
