"""Cell-level bbox annotation for 4x4 detection data (KISS CLI).

Walks ``collected/full/metadata.jsonl`` (full 4x4 images captured at runtime), shows each
image, and records the human's class + the grid cells (1..16) that contain the object to
``annotations.jsonl``. ``prepare_detection_dataset.py`` turns each selected cell into a
YOLO bounding box. Resumable (already-annotated images are skipped).

Usage::

    python training/annotate_detection_cli.py --collected-dir collected/full --open
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import class_mapping
import click


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _annotated_paths(out_path: Path) -> set[str]:
    return {str(r.get("image_path")) for r in _load_jsonl(out_path) if r.get("image_path")}


def _open_externally(image_path: Path) -> None:
    """Open an image in the OS default viewer (best-effort)."""
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(image_path)])
        elif sys.platform.startswith("win"):
            subprocess.Popen(["cmd", "/c", "start", "", str(image_path)], shell=False)
        else:
            subprocess.Popen(["xdg-open", str(image_path)])
    except OSError as e:
        click.echo(f"  (could not open image: {e})")


def _parse_cells(raw: str) -> list[int]:
    """Parse '1,2,5' -> [1,2,5], keeping only cells in 1..16."""
    cells: list[int] = []
    for part in raw.replace(" ", "").split(","):
        if part.isdigit() and 1 <= int(part) <= 16:
            cells.append(int(part))
    return sorted(set(cells))


@click.command()
@click.option(
    "--collected-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("collected/full"),
    help="Directory with full-image metadata.jsonl + images.",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output annotations.jsonl (default: <collected-dir>/annotations.jsonl).",
)
@click.option("--open", "open_images", is_flag=True, help="Open each image in the OS viewer.")
def main(collected_dir: Path, out: Path | None, open_images: bool) -> None:
    """Annotate full 4x4 images with class + cells; append to annotations.jsonl."""
    metadata_path = collected_dir / "metadata.jsonl"
    out_path = out or (collected_dir / "annotations.jsonl")

    records = _load_jsonl(metadata_path)
    if not records:
        click.echo(f"No metadata at {metadata_path}")
        return

    done = _annotated_paths(out_path)
    pending = [r for r in records if r.get("image_path") and str(r["image_path"]) not in done]
    if not pending:
        click.echo("Nothing to annotate -- all images already done.")
        return

    detection_menu = "  ".join(
        f"[{i}] {name}" for i, name in enumerate(class_mapping.DETECTION_CLASSES)
    )
    click.echo(f"{len(pending)} image(s) to annotate. Writing to {out_path}\n")

    with open(out_path, "a", encoding="utf-8") as fh:
        for idx, record in enumerate(pending, 1):
            image_path = Path(str(record["image_path"]))
            click.echo(
                f"[{idx}/{len(pending)}] {image_path}  (hint keyword={record.get('keyword')})"
            )
            if open_images and image_path.exists():
                _open_externally(image_path)

            click.echo(f"  classes: {detection_menu}   [s]kip  [q]uit")
            choice = click.prompt("class", type=str, default="s").strip().lower()
            if choice in {"q", "quit"}:
                click.echo("Stopped. Progress saved.")
                break
            if choice in {"s", "skip", ""}:
                continue

            if choice.isdigit() and 0 <= int(choice) < len(class_mapping.DETECTION_CLASSES):
                label = class_mapping.DETECTION_CLASSES[int(choice)]
            else:
                try:
                    label = class_mapping.DETECTION_CLASSES[
                        class_mapping.detection_class_id(choice)
                    ]
                except KeyError:
                    click.echo(f"  invalid class: {choice!r} -- skipping")
                    continue

            cells = _parse_cells(click.prompt("cells with object (e.g. 1,2,5)", type=str))
            if not cells:
                click.echo("  no valid cells -- skipping")
                continue

            fh.write(
                json.dumps(
                    {"image_path": str(image_path), "label": label, "cells": cells},
                    ensure_ascii=False,
                )
                + "\n"
            )
            fh.flush()


if __name__ == "__main__":
    main()
