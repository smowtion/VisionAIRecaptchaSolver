"""Phase 2 tests: DataCollector writes PNG tiles + JSONL metadata when enabled."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from vision_ai_recaptcha_solver.collection import DataCollector
from vision_ai_recaptcha_solver.config import SolverConfig
from vision_ai_recaptcha_solver.types import CaptchaType


def _tile(value: int = 0) -> np.ndarray:
    return np.full((100, 100, 3), value, dtype=np.uint8)


def _read_metadata(collect_dir: Path) -> list[dict]:
    meta_file = collect_dir / "metadata.jsonl"
    if not meta_file.exists():
        return []
    return [json.loads(line) for line in meta_file.read_text().splitlines() if line.strip()]


def _make_collector(tmp_path: Path) -> DataCollector:
    config = SolverConfig(
        collect_data=True,
        collect_dir=tmp_path / "collected",
        min_confidence_threshold=0.2,
        conf_threshold=0.7,
    )
    return DataCollector(config)


class TestRecordTile:
    def test_uncertain_tile_written(self, tmp_path: Path) -> None:
        collector = _make_collector(tmp_path)
        collector.record_tile(
            _tile(),
            cell=3,
            confidence=0.5,  # within [0.2, 0.7) -> uncertain
            predicted_class="Car",
            captcha_type=CaptchaType.SELECTION_3X3,
            keyword="cars",
        )

        pngs = list((tmp_path / "collected").rglob("*.png"))
        assert len(pngs) == 1

        rows = _read_metadata(tmp_path / "collected")
        assert len(rows) == 1
        row = rows[0]
        assert row["reason"] == "uncertain"
        assert row["captcha_type"] == "selection_3x3"
        assert row["keyword"] == "cars"
        assert row["predicted_class"] == "Car"
        assert abs(row["confidence"] - 0.5) < 1e-6
        assert Path(row["image_path"]).exists()

    def test_confident_tile_skipped(self, tmp_path: Path) -> None:
        collector = _make_collector(tmp_path)
        collector.record_tile(
            _tile(), cell=1, confidence=0.95, predicted_class="Car",
            captcha_type=CaptchaType.SELECTION_3X3, keyword="cars",
        )
        assert _read_metadata(tmp_path / "collected") == []

    def test_below_min_tile_skipped(self, tmp_path: Path) -> None:
        collector = _make_collector(tmp_path)
        collector.record_tile(
            _tile(), cell=1, confidence=0.05, predicted_class="Car",
            captcha_type=CaptchaType.SELECTION_3X3, keyword="cars",
        )
        assert _read_metadata(tmp_path / "collected") == []

    def test_layout_includes_date_and_type(self, tmp_path: Path) -> None:
        collector = _make_collector(tmp_path)
        collector.record_tile(
            _tile(), cell=2, confidence=0.4, predicted_class="Bus",
            captcha_type=CaptchaType.SELECTION_3X3, keyword="buses",
        )
        png = next((tmp_path / "collected").rglob("*.png"))
        # collected/<date>/<captcha_type>/<file>.png
        assert png.parent.name == "selection_3x3"
        assert png.name.startswith("Bus_0.40_")

    def test_context_supplies_type_and_keyword(self, tmp_path: Path) -> None:
        collector = _make_collector(tmp_path)
        collector.set_context(captcha_type=CaptchaType.DYNAMIC_3X3, keyword="bridges")
        collector.record_tile(_tile(), cell=4, confidence=0.5, predicted_class="Bridge")
        row = _read_metadata(tmp_path / "collected")[0]
        assert row["captcha_type"] == "dynamic_3x3"
        assert row["keyword"] == "bridges"


class TestRecordFailure:
    def test_failed_metadata_written(self, tmp_path: Path) -> None:
        collector = _make_collector(tmp_path)
        collector.record_failure(
            captcha_type=CaptchaType.SQUARE_4X4, keyword="stairs", reason="failed"
        )
        rows = _read_metadata(tmp_path / "collected")
        assert len(rows) == 1
        assert rows[0]["reason"] == "failed"
        assert rows[0]["keyword"] == "stairs"

    def test_unknown_keyword_metadata_written(self, tmp_path: Path) -> None:
        collector = _make_collector(tmp_path)
        collector.record_failure(
            captcha_type=CaptchaType.SELECTION_3X3, keyword=None, reason="unknown_keyword"
        )
        rows = _read_metadata(tmp_path / "collected")
        assert len(rows) == 1
        assert rows[0]["reason"] == "unknown_keyword"

    def test_failure_with_images_saved(self, tmp_path: Path) -> None:
        collector = _make_collector(tmp_path)
        collector.record_failure(
            captcha_type=CaptchaType.SQUARE_4X4,
            keyword="stairs",
            reason="failed",
            images=[_tile(10)],
        )
        pngs = list((tmp_path / "collected").rglob("*.png"))
        assert len(pngs) == 1


class TestDisabledZeroIO:
    def test_disabled_creates_no_dir(self, tmp_path: Path) -> None:
        config = SolverConfig(collect_data=False, collect_dir=tmp_path / "collected")
        collector = DataCollector(config)
        collector.record_tile(
            _tile(), cell=1, confidence=0.5, predicted_class="Car",
            captcha_type=CaptchaType.SELECTION_3X3, keyword="cars",
        )
        collector.record_failure(
            captcha_type=CaptchaType.SELECTION_3X3, keyword="cars", reason="failed"
        )
        assert not (tmp_path / "collected").exists()


class TestAsyncSafe:
    def test_record_from_event_loop(self, tmp_path: Path) -> None:
        collector = _make_collector(tmp_path)

        async def run() -> None:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: collector.record_tile(
                    _tile(), cell=5, confidence=0.5, predicted_class="Car",
                    captcha_type=CaptchaType.SELECTION_3X3, keyword="cars",
                ),
            )

        asyncio.run(run())
        assert len(_read_metadata(tmp_path / "collected")) == 1
