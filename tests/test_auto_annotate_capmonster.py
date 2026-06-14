"""Tests for CapMonster auto-annotation helpers + solve flow (network mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock

import auto_annotate_capmonster as aa
import class_mapping as cm


class TestResolveLabel:
    def test_english_keyword(self) -> None:
        assert aa.resolve_detection_label("stairs") == "stairs"
        assert aa.resolve_detection_label("a bridge") == "bridges"

    def test_non_detection_keyword_returns_none(self) -> None:
        assert aa.resolve_detection_label("cars") is None  # COCO class, not custom-detection
        assert aa.resolve_detection_label("") is None
        assert aa.resolve_detection_label(None) is None

    def test_resolved_labels_are_detection_classes(self) -> None:
        for kw in ("stairs", "crosswalks", "chimneys", "tractors"):
            assert aa.resolve_detection_label(kw) in cm.DETECTION_CLASSES


class TestExtractCells:
    def test_dict_cells_key(self) -> None:
        assert aa.extract_cells({"cells": [0, 5, 9]}) == [0, 5, 9]

    def test_dict_alternate_keys(self) -> None:
        assert aa.extract_cells({"answer": [1, 2]}) == [1, 2]

    def test_bare_list(self) -> None:
        assert aa.extract_cells([3, 4]) == [3, 4]

    def test_boolean_mask(self) -> None:
        # CapMonster's real format: 16-element bool mask -> True indices (0-indexed).
        mask = [False] * 16
        for i in (4, 8, 9, 10):
            mask[i] = True
        assert aa.extract_cells({"answer": mask}) == [4, 8, 9, 10]

    def test_empty(self) -> None:
        assert aa.extract_cells({}) == []
        assert aa.extract_cells(None) == []


class TestToOneIndexed:
    def test_shift_and_filter(self) -> None:
        # 0-indexed 0,5,15 -> 1-indexed 1,6,16; out-of-range dropped
        assert aa.to_one_indexed([0, 5, 15, 16, -1]) == [1, 6, 16]

    def test_dedup_sorted(self) -> None:
        assert aa.to_one_indexed([5, 5, 0]) == [1, 6]


class TestSolveImage:
    def test_solve_image_happy_path(self) -> None:
        session = MagicMock()
        create_resp = MagicMock()
        create_resp.json.return_value = {"errorId": 0, "taskId": 123}
        result_resp = MagicMock()
        result_resp.json.return_value = {
            "errorId": 0,
            "status": "ready",
            "solution": {"cells": [0, 5, 9]},
        }
        session.post.side_effect = [create_resp, result_resp]

        cells = aa.solve_image(
            "KEY", "b64data", "Select all squares with stairs", session=session, poll_interval=0
        )
        assert cells == [1, 6, 10]  # 0-indexed -> 1-indexed

    def test_solve_image_create_error_raises(self) -> None:
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {"errorId": 1, "errorCode": "ERROR_KEY_DOES_NOT_EXIST"}
        session.post.return_value = resp

        import pytest

        with pytest.raises(RuntimeError, match="createTask error"):
            aa.solve_image("BAD", "b64", "task", session=session, poll_interval=0)
