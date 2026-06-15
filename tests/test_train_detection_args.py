"""Tier B Phase 3 dry tests: detection trainer + model card + collect driver (no GPU)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from PIL import Image


def test_train_detection_imports_and_defaults() -> None:
    import train_detection

    sig = inspect.signature(train_detection.train)
    assert sig.parameters["imgsz"].default == 640
    assert sig.parameters["amp"].default is True
    assert sig.parameters["device"].default is None
    assert train_detection.DEFAULT_BASE_MODEL == "yolo11x.pt"


def test_train_detection_cli_help() -> None:
    import train_detection
    from click.testing import CliRunner

    result = CliRunner().invoke(train_detection.main, ["--help"])
    assert result.exit_code == 0
    assert "data.yaml" in result.output or "data" in result.output


def test_write_model_card_detect(tmp_path: Path) -> None:
    import class_mapping as cm
    import write_model_card

    onnx = tmp_path / "best.onnx"
    onnx.write_bytes(b"fake-onnx-bytes")

    card = write_model_card.build_card(
        onnx, task="detect", epochs=100, imgsz=640, dataset_size=1200, date="2026-06-14"
    )
    assert card["task"] == "detect"
    assert card["classes"] == cm.DETECTION_CLASSES
    assert card["num_classes"] == len(cm.DETECTION_CLASSES)
    assert len(card["sha256"]) == 64
    assert card["dataset_size"] == 1200


def test_write_model_card_cli_writes_sidecar(tmp_path: Path) -> None:
    import write_model_card
    from click.testing import CliRunner

    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"x")
    out = tmp_path / "m.model_card.json"
    result = CliRunner().invoke(
        write_model_card.main,
        ["--onnx", str(onnx), "--task", "detect", "--out", str(out), "--date", "2026-06-14"],
    )
    assert result.exit_code == 0
    data = json.loads(out.read_text())
    assert data["task"] == "detect"


def test_collect_count_helpers(tmp_path: Path) -> None:
    import collect

    cdir = tmp_path / "collected"
    # one full 4x4 image, two per-cell tiles
    import numpy as np

    (cdir / "full" / "2026-06-14").mkdir(parents=True)
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
        cdir / "full" / "2026-06-14" / "stairs_a.png"
    )
    (cdir / "2026-06-14" / "selection_3x3").mkdir(parents=True)
    for n in ("a", "b"):
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
            cdir / "2026-06-14" / "selection_3x3" / f"{n}.png"
        )

    assert collect.count_full_images(cdir) == 1
    assert collect.count_tiles(cdir) == 2


def test_collect_counts_empty(tmp_path: Path) -> None:
    import collect

    assert collect.count_full_images(tmp_path / "nope") == 0
    assert collect.count_tiles(tmp_path / "nope") == 0
