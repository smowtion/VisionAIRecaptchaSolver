"""Compute the SHA256 of an exported ONNX model for ``YOLODetector.MODEL_SHA256``.

The solver verifies the downloaded model against ``MODEL_SHA256`` as a safety gate. After
exporting a new ``.onnx``, run this and paste the digest into the detector before publishing.

Usage::

    python training/compute_sha256.py path/to/model.onnx
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import click


def compute_sha256(path: Path) -> str:
    """Return the lowercase hex SHA256 of a file.

    Args:
        path: File to hash.

    Returns:
        64-character lowercase hex digest.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


@click.command()
@click.argument("path", type=click.Path(path_type=Path))
def main(path: Path) -> None:
    """CLI entry point: print the SHA256 of PATH."""
    digest = compute_sha256(path)
    click.echo(digest)
    click.echo(f'\nPaste into YOLODetector.MODEL_SHA256:\n    MODEL_SHA256 = "{digest}"')


if __name__ == "__main__":
    main()
