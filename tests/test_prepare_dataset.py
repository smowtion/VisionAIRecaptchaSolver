"""Phase 3 tests: prepare_dataset merges reviewed tiles into an ImageFolder dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import prepare_dataset as pd
from PIL import Image


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(path)


def _make_reviewed(tmp_path: Path, n_keep: int = 4) -> Path:
    src = tmp_path / "collected"
    records = []
    for i in range(n_keep):
        img = src / f"car_{i}.png"
        _write_png(img)
        records.append({"image_path": str(img), "label": "Car", "action": "keep"})

    discarded = src / "junk.png"
    _write_png(discarded)
    records.append({"image_path": str(discarded), "label": "Car", "action": "discard"})

    reviewed = tmp_path / "reviewed.jsonl"
    reviewed.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return reviewed


def test_merge_copies_into_class_folders(tmp_path: Path) -> None:
    reviewed = _make_reviewed(tmp_path, n_keep=4)
    dataset = tmp_path / "dataset"

    summary = pd.merge(reviewed=reviewed, dataset=dataset, val_split=0.25, seed=0)

    train_cars = list((dataset / "train" / "Car").glob("*.png"))
    val_cars = list((dataset / "val" / "Car").glob("*.png"))

    assert summary["copied"] == 4
    assert summary["skipped"] == 1
    assert len(train_cars) + len(val_cars) == 4
    assert len(val_cars) == 1  # round(4 * 0.25)


def test_discard_not_copied(tmp_path: Path) -> None:
    reviewed = _make_reviewed(tmp_path, n_keep=2)
    dataset = tmp_path / "dataset"
    pd.merge(reviewed=reviewed, dataset=dataset, val_split=0.5, seed=0)

    all_pngs = list(dataset.rglob("*.png"))
    assert all(p.name != "junk.png" for p in all_pngs)
    assert len(all_pngs) == 2


def test_label_normalized_to_canonical_folder(tmp_path: Path) -> None:
    src = tmp_path / "collected"
    img = src / "t.png"
    _write_png(img)
    reviewed = tmp_path / "reviewed.jsonl"
    reviewed.write_text(
        json.dumps({"image_path": str(img), "label": "traffic light", "action": "keep"}) + "\n"
    )
    dataset = tmp_path / "dataset"
    pd.merge(reviewed=reviewed, dataset=dataset, val_split=0.0, seed=0)

    assert list((dataset / "train" / "Traffic Light").glob("*.png"))


def test_deterministic_split(tmp_path: Path) -> None:
    reviewed = _make_reviewed(tmp_path, n_keep=4)
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    s1 = pd.merge(reviewed=reviewed, dataset=d1, val_split=0.25, seed=42)
    s2 = pd.merge(reviewed=reviewed, dataset=d2, val_split=0.25, seed=42)
    assert s1 == s2
