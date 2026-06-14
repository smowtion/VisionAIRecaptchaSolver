---
phase: 4
title: Integrate custom 4x4 detection model + publish
status: completed
priority: P2
effort: 1d
dependencies:
  - 3
---

# Phase 4: Integrate custom 4x4 detection model + publish

## Overview

Tích hợp detection model custom (7 lớp thiếu) vào path 4x4: khi keyword thuộc lớp custom → dùng model detection custom (ưu tiên hơn per-cell classification fallback). Cấu hình đường dẫn/URL model, verify SHA256, cập nhật mapping, tài liệu publish HF.

## Requirements

- Functional:
  - `SolverConfig`: thêm `custom_detection_model_path: Path|str|None=None` (+ validate, sentinel-an toàn) cho model 4x4 custom (None → không dùng, giữ hành vi hiện tại).
  - `YOLODetector`: load model detection custom (optional, lazy); `CUSTOM_DETECTION_MODEL_URL` + `CUSTOM_DETECTION_SHA256` (verify như classification); `get_custom_detection_class(keyword)` qua `DETECTION_LABEL_TO_ID`; `detect_for_grid_custom(...)`.
  - `SquareCaptchaHandler.solve` thứ tự ưu tiên: (1) COCO nếu có lớp → detection COCO; (2) else nếu custom detection có lớp → detection custom; (3) else per-cell classification fallback. Per-cell vẫn là backstop.
  - `is_supported` 4x4 = COCO OR custom-detection OR classification.
- Non-functional:
  - Không model custom (mặc định) → hành vi y hệt sau plan 0805 (per-cell fallback). Public API `__all__` không đổi (config field nội bộ, không export class mới).
  - sync+async đối xứng nếu đụng init detector (cả hai khởi tạo `YOLODetector`).
  - Docs publish HF (checklist) như model classification.

## Architecture

- `config.py`: field mới + validate path (pattern như `model_path`).
- `yolo_detector.py`: nhánh model detection custom song song COCO; SHA256 verify; mapping qua class_mapping.DETECTION_*.
- `square_handler.py`: 3 tầng ưu tiên (COCO → custom detect → per-cell). Truyền cả classification target_class (fallback) — đã có.
- `docs/training-and-flywheel.md`: thêm mục "Custom 4x4 detection model" (train→export→sha256→config/URL→publish→auto-download).

## Related Code Files

- Modify: `src/vision_ai_recaptcha_solver/config.py` (custom_detection_model_path)
- Modify: `src/vision_ai_recaptcha_solver/detector/yolo_detector.py` (load + SHA256 + get_custom_detection_class + detect_for_grid_custom)
- Modify: `src/vision_ai_recaptcha_solver/captcha/square_handler.py` (3-tier priority)
- Modify: `src/vision_ai_recaptcha_solver/solver.py`, `async_solver.py` (truyền config field vào detector — đối xứng)
- Modify: `training/class_mapping.py` (DETECTION_LABEL_TO_ID dùng chung)
- Create: `tests/test_custom_detection_integration.py`
- Modify: `docs/training-and-flywheel.md`, `docs/codebase-summary.md`

## Implementation Steps (TDD)

1. **Test trước:** `test_custom_detection_integration.py` (mock detector): square handler ưu tiên custom detection khi COCO None + custom có lớp; rớt về per-cell khi cả COCO+custom None; `is_supported` 4x4 phản ánh 3 nguồn. Config: `custom_detection_model_path=None` mặc định + validate path sai → raise.
2. `config.py` field + validate.
3. `yolo_detector.py`: load custom detect (lazy/optional) + SHA256 + mapping + detect_for_grid_custom.
4. `square_handler.py`: 3-tier priority.
5. Truyền field vào detector ở `solver.py`/`async_solver.py` (đối xứng).
6. Docs publish HF + flow.
7. ruff + mypy + pytest.

## Success Criteria

- [ ] Test 3-tier priority + config viết TRƯỚC và xanh.
- [ ] Có model custom → 4x4 lớp custom dùng detection; không có → per-cell fallback (hành vi cũ).
- [ ] SHA256 verify chặn model custom không khớp.
- [ ] Public API `__all__` không đổi; sync+async đối xứng; ruff+mypy+pytest clean.
- [ ] Docs publish/flow đầy đủ, lặp lại được.

## Risk Assessment

- **Tăng thời gian load (2 detection model):** lazy-load custom; chỉ khi config set.
- **Quên cập nhật URL/SHA256:** checklist publish + verify SHA256 chặn.
- **Lệch class id custom vs mapping:** class_mapping.DETECTION_* dùng chung train + runtime; test đối chiếu.
- **Regression path 4x4 hiện có:** mặc định None → no-op; test 3-tier + integration (plan trước) bảo vệ.
