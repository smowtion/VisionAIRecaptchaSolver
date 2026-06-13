"""Human-in-the-loop labeling queue for collected captcha tiles (KISS CLI).

reCAPTCHA only returns pass/fail, so collected tiles have no ground-truth per-tile label.
This CLI walks ``collected/metadata.jsonl``, shows each tile (path / optional external
viewer), and records a human decision (class label / skip / discard) to
``reviewed.jsonl``, which ``prepare_dataset.py`` then consumes.

Usage::

    python training/review_cli.py --collected-dir collected --open
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
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _reviewed_image_paths(reviewed_path: Path) -> set[str]:
    return {str(r.get("image_path")) for r in _load_jsonl(reviewed_path) if r.get("image_path")}


def _open_externally(image_path: Path) -> None:
    """Open an image in the OS default viewer (best-effort, never fatal)."""
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(image_path)])
        elif sys.platform.startswith("win"):
            subprocess.Popen(["cmd", "/c", "start", "", str(image_path)], shell=False)
        else:
            subprocess.Popen(["xdg-open", str(image_path)])
    except OSError as e:
        click.echo(f"  (could not open image: {e})")


def _prompt_label() -> str | None:
    """Prompt for a class folder, 'skip', 'discard', or 'quit'. Returns action string."""
    menu = "  ".join(f"[{i}] {name}" for i, name in enumerate(class_mapping.FOLDER_ORDER))
    click.echo(menu)
    click.echo("  [s] skip   [d] discard   [q] quit")

    while True:
        choice = click.prompt("label", type=str, default="s").strip().lower()
        if choice in {"q", "quit"}:
            return "quit"
        if choice in {"s", "skip", ""}:
            return "skip"
        if choice in {"d", "discard"}:
            return "discard"
        if choice.isdigit() and 0 <= int(choice) < len(class_mapping.FOLDER_ORDER):
            return class_mapping.FOLDER_ORDER[int(choice)]
        try:
            return class_mapping.normalize_folder(choice)
        except KeyError:
            click.echo(f"  invalid: {choice!r} -- pick a number, class name, s/d/q")


@click.command()
@click.option(
    "--collected-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("collected"),
    help="Directory containing metadata.jsonl and tile images.",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output reviewed.jsonl (default: <collected-dir>/reviewed.jsonl).",
)
@click.option("--open", "open_images", is_flag=True, help="Open each tile in the OS viewer.")
def main(collected_dir: Path, out: Path | None, open_images: bool) -> None:
    """Review unlabeled collected tiles and append decisions to reviewed.jsonl."""
    metadata_path = collected_dir / "metadata.jsonl"
    reviewed_path = out or (collected_dir / "reviewed.jsonl")

    records = _load_jsonl(metadata_path)
    if not records:
        click.echo(f"No metadata found at {metadata_path}")
        return

    already = _reviewed_image_paths(reviewed_path)
    pending = [r for r in records if r.get("image_path") and str(r["image_path"]) not in already]

    if not pending:
        click.echo("Nothing to review -- all samples already labeled.")
        return

    click.echo(f"{len(pending)} sample(s) to review. Writing to {reviewed_path}\n")

    with open(reviewed_path, "a", encoding="utf-8") as fh:
        for idx, record in enumerate(pending, 1):
            image_path = Path(str(record["image_path"]))
            click.echo(
                f"[{idx}/{len(pending)}] {image_path}  "
                f"(reason={record.get('reason')}, pred={record.get('predicted_class')}, "
                f"conf={record.get('confidence')})"
            )
            if open_images and image_path.exists():
                _open_externally(image_path)

            action = _prompt_label()
            if action == "quit":
                click.echo("Stopped. Progress saved.")
                break

            decision = {
                "image_path": str(image_path),
                "label": action if action not in {"skip", "discard"} else None,
                "action": "keep" if action not in {"skip", "discard"} else action,
                "reason": record.get("reason"),
            }
            fh.write(json.dumps(decision, ensure_ascii=False) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
