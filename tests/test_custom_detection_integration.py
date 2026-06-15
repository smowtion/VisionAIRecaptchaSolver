"""Tier B Phase 4 tests: custom 4x4 detection model integration (3-tier priority).

Default (no custom model) must be a no-op: 4x4 keeps COCO + per-cell fallback behavior.
When a custom detection model is present, 4x4 classes the COCO model lacks use it.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from vision_ai_recaptcha_solver.captcha import square_handler as sh_module
from vision_ai_recaptcha_solver.captcha.square_handler import SquareCaptchaHandler
from vision_ai_recaptcha_solver.config import SolverConfig


def _make_handler(detector: MagicMock) -> SquareCaptchaHandler:
    handler = SquareCaptchaHandler(detector, SolverConfig(conf_threshold=0.7), logger=MagicMock())
    handler.get_image_urls = MagicMock(return_value=["http://x/img.png"])  # type: ignore[method-assign]
    handler.download_main_image = MagicMock(  # type: ignore[method-assign]
        return_value=(None, np.zeros((450, 450, 3), dtype=np.uint8))
    )
    handler.click_cells = MagicMock()  # type: ignore[method-assign]
    handler.human_delay = MagicMock()  # type: ignore[method-assign]
    return handler


class TestConfig:
    def test_custom_detection_model_path_defaults_none(self) -> None:
        assert SolverConfig().custom_detection_model_path is None

    def test_custom_detection_model_path_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError, match="custom_detection_model_path must be a str or Path"):
            SolverConfig(custom_detection_model_path=123)  # type: ignore[arg-type]


class TestTypesMapping:
    def test_custom_detection_mapping_covers_seven_classes(self) -> None:
        from vision_ai_recaptcha_solver.types import CUSTOM_DETECTION_TARGET_MAPPINGS

        for kw in ["stairs", "bridges", "crosswalks", "chimneys", "tractors", "palm trees"]:
            assert kw in CUSTOM_DETECTION_TARGET_MAPPINGS

    def test_custom_detection_matches_training_classes(self) -> None:
        # Runtime mapping (types) must agree with the training class order.
        import class_mapping as cm

        from vision_ai_recaptcha_solver.types import CUSTOM_DETECTION_CLASSES

        assert CUSTOM_DETECTION_CLASSES == cm.DETECTION_CLASSES


class TestThreeTierPriority:
    def setup_method(self) -> None:
        sh_module.get_target_keyword = lambda browser: "stairs"  # type: ignore[assignment]

    def test_custom_detection_used_when_coco_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sh_module, "get_target_keyword", lambda browser: "stairs")
        detector = MagicMock()
        detector.get_coco_target_class.return_value = None
        detector.has_custom_detection = True
        detector.get_custom_detection_class.return_value = 5
        detector.detect_for_grid_custom.return_value = [2, 7]

        handler = _make_handler(detector)
        result = handler.solve(browser=MagicMock(), target_class=11)

        detector.detect_for_grid_custom.assert_called_once()
        detector.classify_tiles_with_confidence.assert_not_called()
        assert sorted(result) == [2, 7]

    def test_falls_back_to_per_cell_when_no_custom(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sh_module, "get_target_keyword", lambda browser: "stairs")
        detector = MagicMock()
        detector.get_coco_target_class.return_value = None
        detector.has_custom_detection = False
        detector.classify_tiles_with_confidence.return_value = [(i + 1, 0.9 if i == 0 else 0.1) for i in range(16)]

        handler = _make_handler(detector)
        result = handler.solve(browser=MagicMock(), target_class=11)

        detector.classify_tiles_with_confidence.assert_called_once()
        detector.detect_for_grid_custom.assert_not_called()
        assert result == [1]

    def test_coco_still_primary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sh_module, "get_target_keyword", lambda browser: "cars")
        detector = MagicMock()
        detector.get_coco_target_class.return_value = 2
        detector.has_custom_detection = True
        detector.detect_for_grid.return_value = [3]

        handler = _make_handler(detector)
        result = handler.solve(browser=MagicMock(), target_class=3)

        detector.detect_for_grid.assert_called_once()
        detector.detect_for_grid_custom.assert_not_called()
        detector.classify_tiles_with_confidence.assert_not_called()
        assert result == [3]


class TestIsSupportedWithCustom:
    def test_is_supported_4x4_via_classification_or_custom(self) -> None:
        import logging

        from vision_ai_recaptcha_solver.detector.yolo_detector import YOLODetector
        from vision_ai_recaptcha_solver.types import CaptchaType

        det = object.__new__(YOLODetector)
        det.logger = logging.getLogger("test")
        det._class_names = {}
        det._executor = None
        # stairs: not COCO, but classification covers it -> supported
        assert det.is_supported("stairs", CaptchaType.SQUARE_4X4) is True
