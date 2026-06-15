"""Train the reCAPTCHA tile classifier (YOLO classification) on the merged dataset.

Converted from ``train_model.ipynb``. The heavy ``ultralytics`` import is deferred into
``train()`` so the module imports and ``--help`` work on any machine. Hyperparameters
mirror the original notebook.

Device is auto-detected (CUDA > MPS > CPU), so this runs on a cloud CUDA GPU, an Apple
Silicon Mac (Metal/MPS), or CPU. On MPS, pass ``--no-amp`` if mixed precision misbehaves.

Usage::

    python training/train.py --data training/dataset                 # auto device
    python training/train.py --data training/dataset --device mps --no-amp   # Apple Silicon
    python training/train.py --device 0                              # explicit CUDA
    python training/train.py --resume                                # continue from best.pt
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
from device_utils import resolve_device

DEFAULT_BASE_MODEL = "yolo11x-cls.pt"
DEFAULT_PROJECT = "runs/classify"
DEFAULT_NAME = "rec_cls_model"


def train(
    data: str = "training/dataset",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 64,
    device: int | str | None = None,
    workers: int = 8,
    patience: int = 15,
    base_model: str = DEFAULT_BASE_MODEL,
    project: str = DEFAULT_PROJECT,
    name: str = DEFAULT_NAME,
    resume: bool = False,
    amp: bool = True,
) -> Any:
    """Train (or resume) the classification model. Requires ultralytics at runtime.

    Args:
        data: ImageFolder dataset root (``<data>/train``, ``<data>/val``).
        epochs: Training epochs.
        imgsz: Input image size.
        batch: Batch size.
        device: CUDA index / "mps" / "cpu". None or "auto" auto-detects (CUDA > MPS > CPU).
        workers: Dataloader workers.
        patience: Early-stopping patience.
        base_model: Pretrained base weights to fine-tune from.
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
        cache=True,
    )


@click.command()
@click.option("--data", default="training/dataset", help="ImageFolder dataset root.")
@click.option("--epochs", default=50, type=int, help="Training epochs.")
@click.option("--imgsz", default=640, type=int, help="Input image size.")
@click.option("--batch", default=64, type=int, help="Batch size.")
@click.option("--device", default="auto", help="CUDA index / 'mps' / 'cpu' / 'auto' (default).")
@click.option("--workers", default=8, type=int, help="Dataloader workers.")
@click.option("--patience", default=15, type=int, help="Early-stopping patience.")
@click.option("--base-model", default=DEFAULT_BASE_MODEL, help="Pretrained base weights.")
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
    """CLI entry point: train or resume the classifier (auto-detects CUDA/MPS/CPU)."""
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
