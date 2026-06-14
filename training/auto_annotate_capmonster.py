"""Auto-label collected 4x4 images via CapMonster ComplexImageTask (image mode, ~$0.04/1k).

Breaks the human-annotation bottleneck for the Tier B detection dataset: for each full 4x4
image in ``collected/full/metadata.jsonl``, ask CapMonster which grid cells contain the
target, and write an ``annotations.jsonl`` compatible with ``prepare_detection_dataset.py``.

CapMonster returns 0-indexed cells (0..15); we store 1-indexed (1..16) to match the dataset
builder. Only images whose keyword maps to one of the 7 detection classes are labeled.

API key: ``--api-key`` or env ``CAPMONSTER_API_KEY``. Cost: ~$0.04 / 1000 images.

Usage::

    export CAPMONSTER_API_KEY=...
    python training/auto_annotate_capmonster.py --collected-dir collected/full
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

import click
import requests

from vision_ai_recaptcha_solver.types import (
    CUSTOM_DETECTION_CLASSES,
    CUSTOM_DETECTION_TARGET_MAPPINGS,
)

CREATE_URL = "https://api.capmonster.cloud/createTask"
RESULT_URL = "https://api.capmonster.cloud/getTaskResult"
_SOLUTION_KEYS = ("cells", "answer", "answers", "numbers", "coordinates")


def resolve_detection_label(keyword: str | None) -> str | None:
    """Map a (possibly multilingual) challenge keyword to a detection class label, or None."""
    if not keyword:
        return None
    kw = keyword.lower()
    for key, idx in CUSTOM_DETECTION_TARGET_MAPPINGS.items():
        if key in kw:
            return CUSTOM_DETECTION_CLASSES[idx]
    return None


def extract_cells(solution: Any) -> list[int]:
    """Return 0-indexed selected cells from CapMonster's solution.

    CapMonster ComplexImageTask returns ``solution.answer`` as a 16-element BOOLEAN mask
    (one flag per cell, True = contains the target). We also defensively handle a plain
    list of cell indices in case the format varies.
    """
    if isinstance(solution, list):
        raw = solution
    elif isinstance(solution, dict):
        raw = next((solution[k] for k in _SOLUTION_KEYS if isinstance(solution.get(k), list)), [])
    else:
        raw = []
    if not raw:
        return []
    # Boolean mask -> indices where True. (bool is a subclass of int, so check bool first.)
    if all(isinstance(x, bool) for x in raw):
        return [i for i, flag in enumerate(raw) if flag]
    # Already a list of cell indices.
    return [int(x) for x in raw]


def to_one_indexed(cells: list[int], grid: int = 4) -> list[int]:
    """Convert CapMonster 0-indexed cells (0..15) to dataset 1-indexed (1..16)."""
    n = grid * grid
    return sorted({c + 1 for c in cells if 0 <= c < n})


def solve_image(
    api_key: str,
    image_b64: str,
    task_text: str,
    grid: str = "4x4",
    *,
    session: requests.Session | None = None,
    poll_interval: float = 3.0,
    timeout: float = 120.0,
) -> list[int]:
    """Solve one grid image via CapMonster; return 1-indexed cells. Raises on failure."""
    s = session or requests.Session()
    create = s.post(
        CREATE_URL,
        json={
            "clientKey": api_key,
            "task": {
                "type": "ComplexImageTask",
                "class": "recaptcha",
                "imagesBase64": [image_b64],
                "metadata": {"Task": task_text, "Grid": grid},
            },
        },
        timeout=30,
    ).json()
    if create.get("errorId"):
        raise RuntimeError(
            f"createTask error: {create.get('errorCode')} {create.get('errorDescription')}"
        )
    task_id = create["taskId"]

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(poll_interval)
        res = s.post(RESULT_URL, json={"clientKey": api_key, "taskId": task_id}, timeout=30).json()
        if res.get("errorId"):
            raise RuntimeError(f"getTaskResult error: {res.get('errorCode')}")
        if res.get("status") == "ready":
            grid_n = int(grid.split("x")[0])
            return to_one_indexed(extract_cells(res.get("solution", {})), grid_n)
    raise TimeoutError(f"task {task_id} not ready within {timeout}s")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@click.command()
@click.option(
    "--collected-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("collected/full"),
    help="Directory with full-image metadata.jsonl + images.",
)
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--api-key", default=None, help="CapMonster key (or env CAPMONSTER_API_KEY).")
def main(collected_dir: Path, out: Path | None, api_key: str | None) -> None:
    """Auto-label collected 4x4 images via CapMonster -> annotations.jsonl."""
    api_key = api_key or os.environ.get("CAPMONSTER_API_KEY")
    if not api_key:
        raise click.ClickException("Set --api-key or env CAPMONSTER_API_KEY")

    out_path = out or (collected_dir / "annotations.jsonl")
    done = {str(r.get("image_path")) for r in _load_jsonl(out_path)}
    records = _load_jsonl(collected_dir / "metadata.jsonl")
    pending = [r for r in records if str(r.get("image_path")) not in done]

    session = requests.Session()
    labeled = skipped = errors = 0
    with open(out_path, "a", encoding="utf-8") as fh:
        for i, rec in enumerate(pending, 1):
            image_path = Path(str(rec.get("image_path")))
            label = resolve_detection_label(rec.get("keyword"))
            if label is None or not image_path.exists():
                skipped += 1
                continue
            try:
                b64 = base64.b64encode(image_path.read_bytes()).decode()
                cells = solve_image(
                    api_key, b64, f"Select all squares with {label}", session=session
                )
            except Exception as e:  # network/timeout/parse -> skip this image, keep going
                errors += 1
                click.echo(f"  [{i}/{len(pending)}] {image_path.name}: {type(e).__name__}: {e}")
                continue
            if not cells:
                skipped += 1
                continue
            fh.write(
                json.dumps(
                    {
                        "image_path": str(image_path),
                        "label": label,
                        "cells": cells,
                        "action": "keep",
                        "source": "capmonster",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fh.flush()
            labeled += 1
            click.echo(f"  [{i}/{len(pending)}] {image_path.name} -> {label} cells={cells}")

    cost = labeled * 0.04 / 1000
    click.echo(
        f"\nLabeled {labeled}, skipped {skipped}, errors {errors}. "
        f"Est. CapMonster cost ~${cost:.4f} -> {out_path}"
    )


if __name__ == "__main__":
    main()
