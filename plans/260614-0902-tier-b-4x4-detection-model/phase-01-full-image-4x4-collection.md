---
phase: 1
title: Full-image 4x4 collection
status: completed
priority: P1
effort: 0.5-1d
dependencies: []
---

# Phase 1: Full-image 4x4 collection

## Overview

Thêm nhánh thu **ảnh 4x4 full** (cho detection dataset) vào `DataCollector`, song song nhánh per-cell classification hiện có. Khi `collect_data=True` và gặp 4x4, lưu ảnh challenge nguyên (450x450) + metadata (keyword, captcha_type) vào `collected/full/` để annotate bbox sau.

## Requirements

- Functional:
  - `DataCollector.record_challenge_image(image, keyword, captcha_type, reason="detection_4x4")`: lưu PNG ảnh full + 1 dòng JSONL (`collected/full/metadata.jsonl`) schema `{ts, captcha_type, keyword, image_path, reason}`. No-op khi `collect_data=False`.
  - Hook: `SquareCaptchaHandler` (hoặc solver path 4x4) gọi `record_challenge_image(main_image, keyword, SQUARE_4X4)` khi collector bật — ưu tiên thu khi keyword là lớp COCO-thiếu (7 lớp) để gom đúng data cần.
  - Layout: `collected/full/{YYYY-MM-DD}/{keyword}_{uuid8}.png` + `collected/full/metadata.jsonl`.
- Non-functional:
  - Per-cell collection cũ GIỮ NGUYÊN (không phá). Best-effort, không raise vào solve.
  - Tắt → zero I/O. Lock thread-safe (tái dùng pattern collector).

## Architecture

- `collection/collector.py`: thêm `record_challenge_image` (tái dùng `_save_image`/`_append_metadata` nhưng dir `full/`; cân nhắc tách `_metadata_path(subdir)`); `except Exception` (best-effort).
- Hook đặt ở `square_handler.solve` sau `download_main_image` (đã có `main_image`), trước detection — chỉ thu, không đổi logic giải. Handler nhận collector qua detector (đã có `self.detector.collector`) hoặc inject; KISS: dùng `self.detector.collector`.
- Chỉ thu 4x4 (detection dataset là cho 4x4).

## Related Code Files

- Modify: `src/vision_ai_recaptcha_solver/collection/collector.py` (record_challenge_image)
- Modify: `src/vision_ai_recaptcha_solver/captcha/square_handler.py` (hook thu ảnh full)
- Modify: `tests/test_data_collector.py` (test record_challenge_image)

## Implementation Steps (TDD)

1. **Test trước:** `test_data_collector.py` — `collect_data=True` → `record_challenge_image` tạo PNG dưới `collected/full/<date>/` + 1 dòng JSONL đúng schema; `collect_data=False` → no file.
2. Hiện thực `record_challenge_image` trong collector.
3. Cắm hook vào `square_handler.solve` (qua `self.detector.collector`, guard enabled).
4. ruff + mypy + pytest.

## Success Criteria

- [ ] Test `record_challenge_image` (bật/tắt) viết TRƯỚC và xanh.
- [ ] 4x4 + collect bật → ảnh full + metadata vào `collected/full/`.
- [ ] Per-cell collection cũ không đổi; tắt → zero I/O.
- [ ] ruff + mypy `src/` clean; public API không đổi.

## Risk Assessment

- **Trùng/loãng data:** ưu tiên thu 7 lớp COCO-thiếu; doc khuyến nghị.
- **Phá collector cũ:** thêm method mới, không sửa `record_tile`; test cả hai.
- **Đụng anti-bot do I/O:** ghi nền, best-effort, đã có lock.
