"""Phase 1 scaffold tests: DataCollector is a safe no-op when disabled."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vision_ai_recaptcha_solver.collection import DataCollector
from vision_ai_recaptcha_solver.config import SolverConfig
from vision_ai_recaptcha_solver.types import CaptchaType


def _tile() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


class TestCollectorDisabled:
    """When collect_data is False the collector must do nothing."""

    def test_enabled_is_false_by_default(self) -> None:
        collector = DataCollector(SolverConfig())
        assert collector.enabled is False

    def test_record_tile_noop_when_disabled(self, tmp_path: Path) -> None:
        config = SolverConfig(collect_dir=tmp_path / "collected")
        collector = DataCollector(config)

        # Confidence is inside the uncertain band, but disabled -> no write.
        collector.record_tile(
            _tile(),
            cell=1,
            confidence=0.5,
            predicted_class="Car",
            captcha_type=CaptchaType.SELECTION_3X3,
            keyword="cars",
        )

        assert not (tmp_path / "collected").exists()

    def test_record_failure_noop_when_disabled(self, tmp_path: Path) -> None:
        config = SolverConfig(collect_dir=tmp_path / "collected")
        collector = DataCollector(config)

        collector.record_failure(
            captcha_type=CaptchaType.SQUARE_4X4,
            keyword="stairs",
            reason="failed",
        )

        assert not (tmp_path / "collected").exists()

    def test_disabled_methods_do_not_raise(self) -> None:
        collector = DataCollector(SolverConfig())
        # Should be safe to call repeatedly with no side effects.
        collector.set_context(captcha_type=CaptchaType.DYNAMIC_3X3, keyword="buses")
        collector.record_tile(_tile(), cell=2, confidence=0.3, predicted_class="Bus")
        collector.record_failure(
            captcha_type="selection_3x3", keyword=None, reason="unknown_keyword"
        )
