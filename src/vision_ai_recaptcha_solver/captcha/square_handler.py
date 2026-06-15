"""Handler for 4x4 square captchas using YOLO detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vision_ai_recaptcha_solver.browser.navigation import get_target_keyword
from vision_ai_recaptcha_solver.captcha.base_handler import BaseCaptchaHandler
from vision_ai_recaptcha_solver.types import CaptchaType

if TYPE_CHECKING:
    pass


class SquareCaptchaHandler(BaseCaptchaHandler):
    """Handler for 4x4 square captchas using object detection.

    Uses YOLO detection model to detect all instances of the target object
    across the full captcha image, then selects all grid cells that contain
    the detected objects.
    """

    captcha_type = CaptchaType.SQUARE_4X4

    GRID_SIZE = 450

    GRID_CELLS = 4

    def solve(self, browser: Any, target_class: int) -> list[int]:
        """Solve a square 4x4 captcha.

        Primary path: COCO detection model (best for one large object spanning cells).
        Fallback (when the keyword is not a COCO class, e.g. stairs/bridges/crosswalks):
        per-cell classification with the 57k model, which covers all 14 reCAPTCHA classes.

        Args:
            browser: Browser instance from recaptcha_domain_replicator.
            target_class: Classification class index (used for the per-cell fallback).

        Returns:
            List of cells that were clicked.
        """
        keyword = get_target_keyword(browser)
        if not keyword:
            self.logger.warning("Could not extract target keyword")
            return []

        # Get image URLs and download main image
        img_urls = self.get_image_urls(browser)
        if not img_urls:
            self.logger.warning("No captcha images found")
            return []

        _, main_image = self.download_main_image(img_urls[0])

        # Active-learning hook: capture the full 4x4 image for the detection dataset
        # (separate from per-cell tile collection). No-op unless collection is enabled.
        collector = self.detector.collector
        if collector is not None and collector.enabled:
            collector.record_challenge_image(main_image, keyword, self.captcha_type)

        # 3-tier priority for 4x4:
        #   1) COCO detection (yolo12x) when the class is covered
        #   2) custom Tier-B detection model when loaded and it covers the class
        #   3) per-cell classification fallback (57k model) — always available
        coco_class = self.detector.get_coco_target_class(keyword)
        if coco_class is not None:
            self.logger.debug(f"Target: '{keyword}' -> COCO class {coco_class}")
            answers = self.detector.detect_for_grid(
                main_image,
                target_class=coco_class,
                grid_size=self.GRID_SIZE,
            )
        elif self.detector.has_custom_detection:
            custom_class = self.detector.get_custom_detection_class(keyword)
            if custom_class is not None:
                self.logger.debug(f"Target: '{keyword}' -> custom detection class {custom_class}")
                answers = self.detector.detect_for_grid_custom(
                    main_image,
                    target_class=custom_class,
                    grid_size=self.GRID_SIZE,
                )
            else:
                answers = self._classify_cells_fallback(main_image, keyword, target_class)
        else:
            # No COCO/custom class -> classify each of the 16 cells with the 57k model.
            answers = self._classify_cells_fallback(main_image, keyword, target_class)

        # Filter to valid cell range (1-16)
        valid_answers = [a for a in answers if 1 <= a <= 16]

        if not valid_answers:
            self.logger.info("No targets detected")
            return []

        self.logger.info(f"Targets detected in cells: {valid_answers}")

        self.click_cells(browser, sorted(valid_answers, reverse=True))
        self.human_delay(0.1, 0.2)

        return valid_answers

    def _classify_cells_fallback(
        self, main_image: Any, keyword: str, target_class: int
    ) -> list[int]:
        """Per-cell classification fallback for 4x4 classes the COCO model lacks.

        Splits the image into 16 cells (via the detector's existing tile cropper) and keeps
        cells whose target-class confidence meets ``conf_threshold``.
        """
        if target_class is None or target_class < 0:
            self.logger.critical(
                f"Unknown target for 4x4 fallback: '{keyword}' (no COCO + no classification id)"
            )
            return []

        self.logger.debug(
            f"Target: '{keyword}' not in COCO -> per-cell classification (class {target_class})"
        )
        cell_confidences = self.detector.classify_tiles_with_confidence(
            main_image, self.GRID_CELLS, target_class
        )
        return [cell for cell, conf in cell_confidences if conf >= self.config.conf_threshold]
