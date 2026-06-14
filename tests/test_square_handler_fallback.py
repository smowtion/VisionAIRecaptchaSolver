"""Phase 2 tests: 4x4 square handler falls back to per-cell classification.

When the keyword is not a COCO class, the handler must classify the 16 cells with the
57k classification model (covers all 14 classes) instead of giving up.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from vision_ai_recaptcha_solver.captcha import square_handler as sh_module
from vision_ai_recaptcha_solver.captcha.square_handler import SquareCaptchaHandler
from vision_ai_recaptcha_solver.config import SolverConfig


def _make_handler(detector: MagicMock) -> SquareCaptchaHandler:
    config = SolverConfig(conf_threshold=0.7)
    handler = SquareCaptchaHandler(detector, config, logger=MagicMock())
    # Isolate from real browser/network: stub image fetch + clicks.
    handler.get_image_urls = MagicMock(return_value=["http://x/img.png"])  # type: ignore[method-assign]
    handler.download_main_image = MagicMock(  # type: ignore[method-assign]
        return_value=(None, np.zeros((450, 450, 3), dtype=np.uint8))
    )
    handler.click_cells = MagicMock()  # type: ignore[method-assign]
    handler.human_delay = MagicMock()  # type: ignore[method-assign]
    return handler


@pytest.fixture(autouse=True)
def _patch_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sh_module, "get_target_keyword", lambda browser: "stairs")


def test_fallback_to_per_cell_classification_when_coco_missing() -> None:
    detector = MagicMock()
    detector.get_coco_target_class.return_value = None  # "stairs" not in COCO
    # 16 cells; cells 1 and 5 are above threshold (0.7).
    confs = [(i + 1, 0.9 if i in (0, 4) else 0.1) for i in range(16)]
    detector.classify_tiles_with_confidence.return_value = confs

    handler = _make_handler(detector)
    result = handler.solve(browser=MagicMock(), target_class=11)  # 11 = Stair

    detector.classify_tiles_with_confidence.assert_called_once()
    # signature: classify_tiles_with_confidence(main_image, grid_cells=4, target_class)
    args = detector.classify_tiles_with_confidence.call_args.args
    assert args[1] == 4
    assert args[2] == 11
    assert sorted(result) == [1, 5]
    handler.click_cells.assert_called_once()
    detector.detect_for_grid.assert_not_called()


def test_coco_path_used_when_class_covered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sh_module, "get_target_keyword", lambda browser: "cars")
    detector = MagicMock()
    detector.get_coco_target_class.return_value = 2  # cars in COCO
    detector.detect_for_grid.return_value = [3, 4]

    handler = _make_handler(detector)
    result = handler.solve(browser=MagicMock(), target_class=3)

    detector.detect_for_grid.assert_called_once()
    detector.classify_tiles_with_confidence.assert_not_called()
    assert sorted(result) == [3, 4]
    handler.click_cells.assert_called_once()


def test_fallback_sentinel_target_class_returns_empty() -> None:
    # COCO miss + invalid classification id (-1 sentinel) -> no classification, no clicks.
    detector = MagicMock()
    detector.get_coco_target_class.return_value = None

    handler = _make_handler(detector)
    result = handler.solve(browser=MagicMock(), target_class=-1)

    assert result == []
    detector.classify_tiles_with_confidence.assert_not_called()
    handler.click_cells.assert_not_called()


def test_fallback_no_confident_cells_returns_empty() -> None:
    detector = MagicMock()
    detector.get_coco_target_class.return_value = None
    detector.classify_tiles_with_confidence.return_value = [(i + 1, 0.1) for i in range(16)]

    handler = _make_handler(detector)
    result = handler.solve(browser=MagicMock(), target_class=11)

    assert result == []
    handler.click_cells.assert_not_called()
