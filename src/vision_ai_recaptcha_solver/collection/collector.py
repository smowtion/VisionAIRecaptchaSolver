"""DataCollector: persist uncertain / failed / unknown captcha samples for review.

The collector is the single write point for the active-learning data flywheel. It is
wired into ``YOLODetector`` (tile hook) and the solvers (failure hooks) but stays a
no-op unless ``SolverConfig.collect_data`` is True, so default PyPI usage pays zero cost.

Samples land under ``collect_dir`` as::

    collected/<YYYY-MM-DD>/<captcha_type>/<pred_class>_<conf>_<uuid8>.png
    collected/metadata.jsonl   # one JSON object per saved sample / failure
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

import cv2

from vision_ai_recaptcha_solver.types import CaptchaType

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from vision_ai_recaptcha_solver.config import SolverConfig

_DEFAULT_COLLECT_DIR = Path("collected")
_METADATA_FILENAME = "metadata.jsonl"


def _type_str(captcha_type: CaptchaType | str | None) -> str:
    """Normalize a captcha type (enum / string / None) to a directory-safe string."""
    if captcha_type is None:
        return CaptchaType.UNKNOWN.value
    if isinstance(captcha_type, CaptchaType):
        return captcha_type.value
    return str(captcha_type)


class DataCollector:
    """Persist captcha samples flagged for human review (opt-in).

    All public ``record_*`` methods return immediately when collection is disabled.
    One collector instance is owned per solver, and a solver serializes its own
    detector/solve calls, so there is no concurrent access in normal use. Disk writes
    are still guarded by a lock so a shared ``collect_dir`` (one collector per solver,
    distinct dirs recommended) appends to ``metadata.jsonl`` safely.
    """

    def __init__(self, config: SolverConfig, logger: logging.Logger | None = None) -> None:
        """Initialize the collector from solver config.

        Args:
            config: Solver configuration (reads ``collect_data``, ``collect_dir`` and the
                confidence thresholds that define the "uncertain" band).
            logger: Logger instance. If None, a module logger is used.
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.enabled: bool = bool(config.collect_data)

        collect_dir = config.collect_dir if config.collect_dir is not None else _DEFAULT_COLLECT_DIR
        self.collect_dir: Path = Path(collect_dir)

        self._min_conf = float(config.min_confidence_threshold)
        self._conf = float(config.conf_threshold)

        self._lock = Lock()
        # Per-solve context filled in by the solver, read when the detector forwards tiles.
        self._ctx_captcha_type: CaptchaType | str | None = None
        self._ctx_keyword: str | None = None

    def set_context(
        self,
        captcha_type: CaptchaType | str | None = None,
        keyword: str | None = None,
    ) -> None:
        """Set the current solve context used to annotate forwarded tiles.

        Only provided fields are updated, so the solver can set ``captcha_type`` and
        ``keyword`` at different points in the solve loop.
        """
        if not self.enabled:
            return
        if captcha_type is not None:
            self._ctx_captcha_type = captcha_type
        if keyword is not None:
            self._ctx_keyword = keyword

    def record_tile(
        self,
        image: NDArray[np.uint8],
        cell: int,
        confidence: float,
        *,
        predicted_class: str | None = None,
        captcha_type: CaptchaType | str | None = None,
        keyword: str | None = None,
    ) -> None:
        """Record a single tile if its confidence is in the uncertain band.

        Uncertain band = ``min_confidence_threshold <= confidence < conf_threshold``.
        Confident or clearly-negative tiles are skipped. No-op when disabled.

        Args:
            image: Tile image (BGR numpy array, already cropped by the detector).
            cell: 1-indexed cell number within the grid.
            confidence: Target-class confidence for this tile.
            predicted_class: Class name being searched (the target label).
            captcha_type: Captcha type; falls back to the context if omitted.
            keyword: Challenge keyword; falls back to the context if omitted.
        """
        if not self.enabled:
            return
        if not (self._min_conf <= confidence < self._conf):
            return

        ctype = _type_str(captcha_type if captcha_type is not None else self._ctx_captcha_type)
        kw = keyword if keyword is not None else self._ctx_keyword

        try:
            with self._lock:
                image_path = self._save_image(image, predicted_class, confidence, ctype, cell)
                self._append_metadata(
                    {
                        "captcha_type": ctype,
                        "keyword": kw,
                        "predicted_class": predicted_class,
                        "confidence": round(float(confidence), 4),
                        "reason": "uncertain",
                        "image_path": str(image_path),
                        "solve_outcome": "pending",
                    }
                )
        except Exception as e:
            # Collector is best-effort telemetry; never let it abort a solve.
            self.logger.debug("DataCollector: failed to record tile: %s", e)

    def record_failure(
        self,
        captcha_type: CaptchaType | str | None,
        keyword: str | None,
        reason: str,
        images: list[NDArray[np.uint8]] | None = None,
    ) -> None:
        """Record a solve failure (``failed`` or ``unknown_keyword``). No-op when disabled.

        Args:
            captcha_type: Captcha type at failure time.
            keyword: Challenge keyword (may be None for ``unknown_keyword``).
            reason: One of ``failed`` | ``unknown_keyword``.
            images: Optional challenge images to persist for review.
        """
        if not self.enabled:
            return

        ctype = _type_str(captcha_type)
        try:
            with self._lock:
                for image in images or []:
                    image_path = self._save_image(image, "unknown", 0.0, ctype, cell=None)
                    self._append_metadata(
                        {
                            "captcha_type": ctype,
                            "keyword": keyword,
                            "predicted_class": None,
                            "confidence": None,
                            "reason": reason,
                            "image_path": str(image_path),
                            "solve_outcome": "failed",
                        }
                    )
                if not images:
                    self._append_metadata(
                        {
                            "captcha_type": ctype,
                            "keyword": keyword,
                            "predicted_class": None,
                            "confidence": None,
                            "reason": reason,
                            "image_path": None,
                            "solve_outcome": "failed",
                        }
                    )
        except Exception as e:
            # Collector is best-effort telemetry; never let it abort a solve.
            self.logger.debug("DataCollector: failed to record failure: %s", e)

    def record_challenge_image(
        self,
        image: NDArray[np.uint8],
        keyword: str | None,
        captcha_type: CaptchaType | str | None,
        reason: str = "detection_4x4",
    ) -> None:
        """Save a full (uncropped) challenge image for the detection dataset.

        Separate from ``record_tile`` (per-cell classification): this captures the whole
        4x4 image + metadata under ``<collect_dir>/full/`` so it can be bbox-annotated
        later to train a detection model. No-op when disabled; best-effort (never raises
        into the solve loop).
        """
        if not self.enabled:
            return

        ctype = _type_str(captcha_type)
        try:
            full_dir = self.collect_dir / "full"
            with self._lock:
                day_dir = full_dir / date.today().isoformat()
                day_dir.mkdir(parents=True, exist_ok=True)
                label = (keyword or "unknown").replace(" ", "-")
                image_path = day_dir / f"{label}_{uuid.uuid4().hex[:8]}.png"
                cv2.imwrite(str(image_path), image)
                self._append_metadata(
                    {
                        "captcha_type": ctype,
                        "keyword": keyword,
                        "reason": reason,
                        "image_path": str(image_path),
                    },
                    subdir="full",
                )
        except Exception as e:
            # Collector is best-effort telemetry; never let it abort a solve.
            self.logger.debug("DataCollector: failed to record challenge image: %s", e)

    def _save_image(
        self,
        image: NDArray[np.uint8],
        predicted_class: str | None,
        confidence: float,
        captcha_type: str,
        cell: int | None,
    ) -> Path:
        """Write a tile/challenge image and return its path."""
        day_dir = self.collect_dir / date.today().isoformat() / captcha_type
        day_dir.mkdir(parents=True, exist_ok=True)

        label = (predicted_class or "unknown").replace(" ", "-")
        uid = uuid.uuid4().hex[:8]
        filename = f"{label}_{confidence:.2f}_{uid}.png"
        image_path = day_dir / filename
        cv2.imwrite(str(image_path), image)
        return image_path

    def _append_metadata(self, record: dict[str, Any], subdir: str | None = None) -> None:
        """Append one JSON record to a metadata.jsonl ledger (root, or a subdir)."""
        target_dir = self.collect_dir / subdir if subdir else self.collect_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        record = {"ts": _utc_timestamp(), **record}
        meta_path = target_dir / _METADATA_FILENAME
        with open(meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _utc_timestamp() -> str:
    """ISO-8601 UTC timestamp for metadata records."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
