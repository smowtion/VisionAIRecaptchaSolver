"""Tier B Phase 2 tests: cell->bbox conversion + YOLO detection dataset builder."""

from __future__ import annotations

import json
from pathlib import Path

import class_mapping as cm
import numpy as np
import prepare_detection_dataset as pdd
import pytest
from PIL import Image


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)).save(path)


class TestCellToYoloBbox:
    def test_top_left_cell(self) -> None:
        assert pdd.cell_to_yolo_bbox(1, grid=4) == (0.125, 0.125, 0.25, 0.25)

    def test_bottom_right_cell(self) -> None:
        cx, cy, w, h = pdd.cell_to_yolo_bbox(16, grid=4)
        assert (cx, cy, w, h) == (0.875, 0.875, 0.25, 0.25)

    def test_center_ish_cell(self) -> None:
        # cell 6 -> idx5 -> row1 col1 -> cx=0.375 cy=0.375
        assert pdd.cell_to_yolo_bbox(6, grid=4) == (0.375, 0.375, 0.25, 0.25)

    def test_invalid_cell_raises(self) -> None:
        with pytest.raises(ValueError):
            pdd.cell_to_yolo_bbox(17, grid=4)


class TestDetectionClassMapping:
    def test_detection_classes_contiguous_unique(self) -> None:
        assert len(set(cm.DETECTION_CLASSES)) == len(cm.DETECTION_CLASSES)
        assert list(cm.DETECTION_LABEL_TO_ID.values()) == list(range(len(cm.DETECTION_CLASSES)))

    def test_detection_class_id_by_label_and_folder(self) -> None:
        assert cm.detection_class_id("stairs") == cm.DETECTION_LABEL_TO_ID["stairs"]
        assert cm.detection_class_id("Stair") == cm.DETECTION_LABEL_TO_ID["stairs"]

    def test_detection_classes_are_coco_gap(self) -> None:
        # All detection classes must be classification-known but NOT COCO classes.
        from vision_ai_recaptcha_solver.types import COCO_TARGET_MAPPINGS, TARGET_MAPPINGS

        for label in cm.DETECTION_CLASSES:
            assert label in TARGET_MAPPINGS
            assert label not in COCO_TARGET_MAPPINGS


class TestPrepareDetectionDataset:
    def _annotations(self, tmp_path: Path) -> Path:
        src = tmp_path / "full"
        records = []
        for i in range(4):
            img = src / f"stairs_{i}.png"
            _write_png(img)
            records.append({"image_path": str(img), "label": "stairs", "cells": [1, 6]})
        ann = tmp_path / "annotations.jsonl"
        ann.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        return ann

    def test_builds_yolo_layout(self, tmp_path: Path) -> None:
        ann = self._annotations(tmp_path)
        dataset = tmp_path / "detection_dataset"
        summary = pdd.prepare(annotations=ann, dataset=dataset, val_split=0.25, seed=0)

        imgs = list((dataset / "images").rglob("*.png"))
        labels = list((dataset / "labels").rglob("*.txt"))
        assert len(imgs) == 4
        assert len(labels) == 4
        assert (dataset / "data.yaml").exists()
        assert summary["images"] == 4
        assert summary["val"] == 1

    def test_label_file_has_correct_bboxes(self, tmp_path: Path) -> None:
        ann = self._annotations(tmp_path)
        dataset = tmp_path / "detection_dataset"
        pdd.prepare(annotations=ann, dataset=dataset, val_split=0.0, seed=0)

        label_file = next((dataset / "labels").rglob("*.txt"))
        lines = label_file.read_text().strip().splitlines()
        assert len(lines) == 2  # two cells -> two boxes
        cls_id, cx, cy, w, h = lines[0].split()
        assert int(cls_id) == cm.DETECTION_LABEL_TO_ID["stairs"]
        assert float(w) == 0.25 and float(h) == 0.25

    def test_data_yaml_names_match_detection_classes(self, tmp_path: Path) -> None:
        import yaml

        ann = self._annotations(tmp_path)
        dataset = tmp_path / "detection_dataset"
        pdd.prepare(annotations=ann, dataset=dataset, val_split=0.0, seed=0)
        data = yaml.safe_load((dataset / "data.yaml").read_text())
        names = data["names"]
        ordered = [names[i] for i in range(len(names))] if isinstance(names, dict) else names
        assert list(ordered) == cm.DETECTION_CLASSES
