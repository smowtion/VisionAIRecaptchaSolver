---
phase: 1
title: Fail-fast & speed (is_supported + skip)
status: completed
priority: P1
effort: 0.5-1d
dependencies: []
---

# Phase 1: Fail-fast & speed (is_supported + skip)

## Overview

Thêm `YOLODetector.is_supported(keyword, captcha_type)` và dùng nó trong vòng solve (cả sync + async) để **bỏ nhanh** challenge không giải được (reload với delay tối thiểu, không đốt full `default_timeout`), thay vì treo ~10 phút. Giữ ngân sách wall-clock; nhiều reload-skip hơn → xác suất trúng challenge giải được tăng.

## Requirements

- Functional:
  - `is_supported(keyword, captcha_type) -> bool`: SQUARE_4X4 → `get_coco_target_class(keyword) is not None`; 3x3 (DYNAMIC/SELECTION) → `get_target_class(keyword) is not None`; keyword rỗng → False. (Phase 2 mở rộng nhánh 4x4.)
  - Solve loop: sau khi xác định `captcha_type` + keyword, nếu `not is_supported` → `record_failure(reason="unknown_keyword")` (đã có) + reload NHANH + `continue` mà KHÔNG tính vào budget click; tách đếm `skips` riêng, cap bằng `max_skips` (vd `max_attempts * 3`) HOẶC wall-clock `timeout`.
  - Path reload/skip: delay tối thiểu (vd `human_delay(0.05, 0.02)`), reload chờ ngắn; KHÔNG đổi delay path click/verify.
- Non-functional:
  - sync + async đối xứng; public exception không đổi; ruff/mypy `src/` clean.
  - Không vòng lặp vô hạn (cap skips + wall-clock).

## Architecture

- `detector/yolo_detector.py`: thêm `is_supported`. Dùng `get_coco_target_class` / `get_target_class` sẵn có (không thêm state).
- `solver.py` + `async_solver.py`: trong vòng `while attempts < max_attempts`:
  - Lấy `captcha_type`; lấy keyword (qua `_get_target_class` đã set context). Trước khi gọi handler, nếu unsupported → fast reload + continue (đếm skip riêng, không tăng "real attempt").
  - Đề xuất: đổi vòng sang quản lý cả `attempts` (real solve) + `skips` (fast reload), điều kiện thoát: `attempts >= max_attempts` hoặc `skips >= max_skips` hoặc wall-clock vượt `timeout`.
- Giữ nguyên path raise `TokenExtractionError` cuối (record_failure "failed" đã có).

## Related Code Files

- Modify: `src/vision_ai_recaptcha_solver/detector/yolo_detector.py` (thêm `is_supported`)
- Modify: `src/vision_ai_recaptcha_solver/solver.py` (fast-skip loop + speed)
- Modify: `src/vision_ai_recaptcha_solver/async_solver.py` (đối xứng; fast reload qua `_run_in_executor`)
- Create: `tests/test_is_supported.py`

## Implementation Steps (TDD)

1. **Test trước:** `tests/test_is_supported.py` — `is_supported` với detector stub (skip model load: `object.__new__(YOLODetector)` + set `_class_names`/logger, hoặc monkeypatch `get_target_class`/`get_coco_target_class`):
   - 4x4 + "cars" (COCO có) → True; 4x4 + "stairs" (COCO không) → False; 3x3 + "stairs" → True; keyword "" → False.
   - Regression: full `pytest` xanh trước khi sửa loop.
2. Hiện thực `is_supported` trong `YOLODetector`.
3. Sửa `solver.py`: tách `skips`/`attempts`, fast-skip unsupported, delay tối thiểu path skip, cap skips + wall-clock.
4. Sửa `async_solver.py` đối xứng (fast reload offload executor).
5. ruff + mypy + pytest.

## Success Criteria

- [ ] `tests/test_is_supported.py` viết TRƯỚC và xanh.
- [ ] Unsupported → fast reload (không đốt full timeout), có cap chống vô hạn.
- [ ] sync + async đối xứng; public exception/`__all__` không đổi.
- [ ] full `pytest` xanh (no regression); ruff + mypy `src/` clean.

## Risk Assessment

- **Reload quá nhiều → reCAPTCHA "try again later":** cap `max_skips` + wall-clock + giữ human_delay path click. Mitigation.
- **Sync/async lệch:** checklist sửa cả hai + test `is_supported` chung.
- **Đổi cấu trúc vòng gây regression im lặng:** test-first + integration ở Phase 3.
