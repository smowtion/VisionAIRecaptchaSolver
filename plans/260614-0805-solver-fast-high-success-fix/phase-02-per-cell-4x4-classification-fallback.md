---
phase: 2
title: Per-cell 4x4 classification fallback
status: completed
priority: P1
effort: 0.5-1d
dependencies:
  - 1
---

# Phase 2: Per-cell 4x4 classification fallback

## Overview

Khi 4x4 có lớp KHÔNG nằm trong COCO (bridges, chimneys, crosswalks, mountains, palm trees, stairs, tractors), thay vì bỏ → **fallback chia 4x4 thành 16 cell, classify từng cell bằng model 57k** (đủ 14 lớp), chọn cell có conf ≥ `conf_threshold`. Phủ 14 lớp cho 4x4 ngay, không cần GPU. Cầu nối tới Tầng B (train).

## Requirements

- Functional:
  - `SquareCaptchaHandler.solve(browser, target_class)`: thử COCO detection trước (nếu `get_coco_target_class(keyword)` có VÀ phát hiện được cell). Nếu COCO không có lớp (None) → fallback per-cell classification dùng `target_class` (classification id solver đã truyền vào) qua `detector.classify_tiles_with_confidence(main_image, grid_size=4, target_class)` → chọn cell conf ≥ `conf_threshold` → click.
  - `target_class` truyền vào solve() là classification id (solver đã tính qua `get_target_class`); square handler hiện bỏ qua nó → giờ dùng cho fallback.
  - Mở rộng `YOLODetector.is_supported` (Phase 1): SQUARE_4X4 supported = `get_coco_target_class(keyword) is not None` **HOẶC** `get_target_class(keyword) is not None`.
- Non-functional:
  - Tái dùng `classify_tiles_with_confidence` (đã crop 16 cell) — DRY; collector hook 4x4 tự kích hoạt (bonus data).
  - Không đổi public API; ruff/mypy clean. async dùng chung handler (chạy trong executor) → không cần sửa async riêng cho fallback, nhưng cập nhật `is_supported` ảnh hưởng cả hai.

## Architecture

- `captcha/square_handler.py`: nhánh quyết định COCO vs per-cell. Per-cell: lọc `[(cell, conf)]` với conf ≥ `self.config.conf_threshold`, `valid in 1..16`, click `sorted(reverse=True)` (giữ pattern hiện có).
- `detector/yolo_detector.py`: cập nhật `is_supported` nhánh 4x4 (OR classification).
- Lưu ý ngưỡng: tái dùng `conf_threshold` (unresolved Q2 trong report — chốt: tái dùng, tinh chỉnh sau theo data).

## Related Code Files

- Modify: `src/vision_ai_recaptcha_solver/captcha/square_handler.py` (fallback path)
- Modify: `src/vision_ai_recaptcha_solver/detector/yolo_detector.py` (`is_supported` 4x4 OR classification)
- Modify: `tests/test_is_supported.py` (4x4 + "stairs" giờ → True)
- Create: `tests/test_square_handler_fallback.py`

## Implementation Steps (TDD)

1. **Test trước:** `tests/test_square_handler_fallback.py` — handler với detector mock + browser mock:
   - COCO None ("stairs") → fallback: `classify_tiles_with_confidence` trả confidences giả cho 16 cell → handler chọn đúng cell ≥ threshold + gọi `click_cells` đúng.
   - COCO có ("cars") → đi path detection cũ (mock `detect_for_grid`), KHÔNG gọi fallback.
   - Mock `get_image_urls`/`download_main_image`/`click_cells` để cô lập.
   - Cập nhật `test_is_supported.py`: 4x4 + "stairs" → True.
2. Hiện thực fallback trong `square_handler.py`.
3. Mở rộng `is_supported` (OR classification).
4. ruff + mypy + pytest.

## Success Criteria

- [ ] Test fallback (COCO-miss → per-cell) + COCO-hit (no fallback) viết TRƯỚC và xanh.
- [ ] `is_supported` 4x4 = COCO OR classification; test cập nhật xanh.
- [ ] 4x4 cho lớp ngoài COCO không còn bị bỏ; click cell theo classification.
- [ ] full `pytest` xanh; ruff + mypy `src/` clean; public API không đổi.

## Risk Assessment

- **Per-cell kém với object lớn trải nhiều cell:** chấp nhận như cầu nối; Tầng B (train) là fix thật; collector thu data 4x4 để cải thiện.
- **Ngưỡng conf không tối ưu cho 4x4:** tái dùng `conf_threshold`, tinh chỉnh sau theo data thu được.
- **target_class id mismatch (classification vs COCO):** square handler dùng đúng classification `target_class` cho fallback (không nhầm COCO id).
