"""Phase 1 tests: YOLODetector.is_supported gates fast-skip of unsolvable challenges.

Built with a model-free detector stub (object.__new__) so tests stay fast and offline.
"""

from __future__ import annotations

import logging

from vision_ai_recaptcha_solver.detector.yolo_detector import YOLODetector
from vision_ai_recaptcha_solver.types import CaptchaType


def _detector_stub() -> YOLODetector:
    """A YOLODetector with no loaded model: get_target_class falls back to TARGET_MAPPINGS."""
    det = object.__new__(YOLODetector)
    det.logger = logging.getLogger("test")
    det._class_names = {}  # forces get_target_class to use the TARGET_MAPPINGS fallback
    det._executor = None  # __del__ -> _cleanup_executor touches this; set to avoid a warning
    return det


class TestIsSupported:
    def test_4x4_covered_by_coco(self) -> None:
        det = _detector_stub()
        assert det.is_supported("cars", CaptchaType.SQUARE_4X4) is True

    def test_4x4_missing_from_coco_but_classification_covers(self) -> None:
        # "stairs" is not a COCO class, but the per-cell classification fallback covers it.
        det = _detector_stub()
        assert det.is_supported("stairs", CaptchaType.SQUARE_4X4) is True

    def test_4x4_unmappable_keyword_unsupported(self) -> None:
        det = _detector_stub()
        assert det.is_supported("zzz-not-a-class", CaptchaType.SQUARE_4X4) is False

    def test_3x3_supported_by_classification(self) -> None:
        det = _detector_stub()
        assert det.is_supported("stairs", CaptchaType.SELECTION_3X3) is True
        assert det.is_supported("bridges", CaptchaType.DYNAMIC_3X3) is True

    def test_empty_keyword_unsupported(self) -> None:
        det = _detector_stub()
        assert det.is_supported("", CaptchaType.SELECTION_3X3) is False
        assert det.is_supported("", CaptchaType.SQUARE_4X4) is False

    def test_unknown_keyword_unsupported(self) -> None:
        det = _detector_stub()
        assert det.is_supported("zzz-not-a-class", CaptchaType.SELECTION_3X3) is False
