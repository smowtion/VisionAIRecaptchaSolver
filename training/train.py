"""Train the reCAPTCHA tile classifier (YOLO classification) on the merged dataset.

Converted from ``train_model.ipynb``. Designed to run on a cloud GPU (Mac has no CUDA):
the heavy ``ultralytics`` import is deferred into ``train()`` so the module imports and
``--help`` work on any machine. Hyperparameters mirror the original notebook.

Usage (cloud GPU)::

    python training/train.py --data training/dataset --device 0
    python training/train.py --resume   # continue from last best.pt
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

DEFAULT_BASE_MODEL = "yolo11x-cls.pt"
DEFAULT_PROJECT = "runs/classify"
DEFAULT_NAME = "rec_cls_model"


def train(
    data: str = "training/dataset",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 64,
    device: int | str = 0,
    workers: int = 8,
    patience: int = 15,
    base_model: str = DEFAULT_BASE_MODEL,
    project: str = DEFAULT_PROJECT,
    name: str = DEFAULT_NAME,
    resume: bool = False,
) -> Any:
    """Train (or resume) the classification model. Requires a GPU + ultralytics at runtime.

    Args:
        data: ImageFolder dataset root (``<data>/train``, ``<data>/val``).
        epochs: Training epochs.
        imgsz: Input image size.
        batch: Batch size.
        device: CUDA device index or "cpu".
        workers: Dataloader workers.
        patience: Early-stopping patience.
        base_model: Pretrained base weights to fine-tune from.
        project: Ultralytics project (run output) directory.
        name: Run name under the project directory.
        resume: Resume from ``<project>/<name>/weights/best.pt``.

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
        device=device,
        workers=workers,
        patience=patience,
        project=project,
        name=name,
        amp=True,
        cache=True,
    )


@click.command()
@click.option("--data", default="training/dataset", help="ImageFolder dataset root.")
@click.option("--epochs", default=50, type=int, help="Training epochs.")
@click.option("--imgsz", default=640, type=int, help="Input image size.")
@click.option("--batch", default=64, type=int, help="Batch size.")
@click.option("--device", default="0", help="CUDA device index or 'cpu'.")
@click.option("--workers", default=8, type=int, help="Dataloader workers.")
@click.option("--patience", default=15, type=int, help="Early-stopping patience.")
@click.option("--base-model", default=DEFAULT_BASE_MODEL, help="Pretrained base weights.")
@click.option("--project", default=DEFAULT_PROJECT, help="Run output directory.")
@click.option("--name", default=DEFAULT_NAME, help="Run name.")
@click.option("--resume", is_flag=True, help="Resume from last best.pt.")
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
) -> None:
    """CLI entry point: train or resume the classifier (needs a GPU at runtime)."""
    dev: int | str = int(device) if device.isdigit() else device
    train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=dev,
        workers=workers,
        patience=patience,
        base_model=base_model,
        project=project,
        name=name,
        resume=resume,
    )


if __name__ == "__main__":
    main()
