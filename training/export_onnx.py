"""Export trained YOLO classification weights (.pt) to ONNX for the solver.

Parametrized from the original hardcoded script: pass ``--weights`` to choose the source
``.pt``. The heavy ``ultralytics`` import is deferred so the module imports / ``--help``
work without a GPU. After export, run ``compute_sha256.py`` on the ``.onnx`` and paste the
digest into ``YOLODetector.MODEL_SHA256``.

Usage::

    python training/export_onnx.py --weights runs/classify/rec_cls_model/weights/best.pt
"""

from __future__ import annotations

from pathlib import Path

import click


def export(weights: Path) -> Path:
    """Export a ``.pt`` checkpoint to ONNX (dynamic axes, fp32).

    Args:
        weights: Path to the trained ``.pt`` weights.

    Returns:
        Path to the exported ``.onnx`` file.

    Raises:
        FileNotFoundError: If the weights file does not exist.
    """
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    output = model.export(format="onnx", half=False, dynamic=True)
    onnx_path = Path(output) if output else weights.with_suffix(".onnx")
    return onnx_path


@click.command()
@click.option(
    "--weights",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to the trained .pt weights to export.",
)
def main(weights: Path) -> None:
    """CLI entry point: export the given weights to ONNX."""
    onnx_path = export(weights)
    click.echo(f"Exported ONNX to {onnx_path}")
    click.echo("Next: python training/compute_sha256.py " + str(onnx_path))


if __name__ == "__main__":
    main()
