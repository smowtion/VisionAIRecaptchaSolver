---
phase: 1
title: "Setup & Stabilize Solver + Collector Scaffold"
status: completed
priority: P1
effort: "1-2d"
dependencies: []
---

# Phase 1: Setup & Stabilize Solver + Collector Scaffold

## Overview

Đảm bảo solver cài/chạy/test ổn trên 3.10–3.12; làm cho failure (4x4 COCO gap, low-conf, keyword unknown) *graceful*; dựng khung collector opt-in **no-op** (config flag + module rỗng an toàn) làm nền cho Phase 2. Chưa thực sự ghi dữ liệu.

## Requirements

- Functional:
  - `pip install -e ".[dev]"` + `import vision_ai_recaptcha_solver` OK trên 3.10/3.11/3.12.
  - Thêm config `collect_data: bool = False`, `collect_dir: Path | str | None = None` vào `SolverConfig` (validate; mặc định tắt).
  - Failure paths không crash thô: keyword unknown / `UnsupportedCaptchaError` / 4x4 không giải được → log rõ ràng + đi tiếp/đếm attempt, cuối cùng raise `TokenExtractionError` như cũ (không đổi public exception).
  - Khung `collection/` tạo `DataCollector` no-op (method `record_*` tồn tại nhưng return ngay khi `collect_data=False`).
- Non-functional:
  - Không thêm dependency mới (dùng cv2/PIL/numpy đã có).
  - Public API `__all__` không đổi. ruff + mypy strict pass.

## Architecture

- `SolverConfig` (config.py): thêm 2 field + validate `collect_dir` (nếu set → Path hợp lệ; tách hoàn toàn khỏi `download_dir` để không bị `cleanup_tmp_on_close` xóa).
- `collection/__init__.py` + `collection/collector.py`: `DataCollector(config, logger)` với API tối thiểu:
  - `enabled: bool` (= `config.collect_data`)
  - `record_tile(image, predicted_class, confidence, captcha_type, keyword, reason)` → no-op nếu disabled.
  - `record_failure(captcha_type, keyword, reason, images=None)` → no-op nếu disabled.
  - Phase 1: thân hàm chỉ guard `if not self.enabled: return` (chưa ghi đĩa — Phase 2 mới ghi).
- Solver wiring: `solver.py` + `async_solver.py` khởi tạo `self._collector = DataCollector(...)`. **Inject vào `YOLODetector`** (tham số optional `collector=None`, default None → back-compat) — đây là tầng có sẵn tile crop + conf (quyết định Validation Session 1). Handlers KHÔNG cần tham số collector.

## Related Code Files

- Modify: `src/vision_ai_recaptcha_solver/config.py` (2 field + validate)
- Modify: `src/vision_ai_recaptcha_solver/solver.py` (init collector, wiring; graceful failure)
- Modify: `src/vision_ai_recaptcha_solver/async_solver.py` (đối xứng)
- Modify: `src/vision_ai_recaptcha_solver/detector/yolo_detector.py` (tham số `collector` optional, default None — wiring scaffold; forward thực hiện ở Phase 2)
- Create: `src/vision_ai_recaptcha_solver/collection/__init__.py`
- Create: `src/vision_ai_recaptcha_solver/collection/collector.py`
- Modify: `.gitignore` (thêm `collected/`)
- Modify: `docs/codebase-summary.md`, `docs/code-standards.md` (ghi nhận module mới — Phase cuối)

## Implementation Steps (TDD)

1. **Test trước (lock hành vi hiện tại):**
   - `tests/test_config.py`: thêm test `collect_data` mặc định False; `collect_dir` mặc định None; set `collect_dir` hợp lệ/không hợp lệ.
   - `tests/test_collector_scaffold.py` (mới): `DataCollector(config).enabled is False` khi tắt; `record_tile/record_failure` không raise và không tạo file khi tắt.
   - Regression: chạy full `pytest` (unit) — phải xanh nguyên trạng trước khi sửa.
2. Thêm field vào `SolverConfig` + validate (giữ sentinel pattern không đổi cho port/download_dir).
3. Tạo `collection/collector.py` no-op (guard `enabled`).
4. Wire collector vào `solver.py` + `async_solver.py`; thêm tham số `collector=None` vào `YOLODetector` (default None, back-compat).
5. Rà các failure path trong vòng solve (cả 2 solver): bắt keyword-None, `UnsupportedCaptchaError`, 4x4 thất bại → log + đếm attempt, không đổi exception công khai.
6. `.gitignore` thêm `collected/`.
7. Chạy `ruff check src/`, `ruff format --check src/`, `mypy src/ --ignore-missing-imports`, `pytest`.
8. (Tùy chọn, không CI) verify cài trên 3.11/3.12 nếu có sẵn interpreter.

## Success Criteria

- [x] Test mới (config + collector scaffold) viết TRƯỚC và xanh.
- [x] `pytest` (unit) xanh; không regression.
- [x] ruff + mypy strict pass.
- [x] `collect_data=False` mặc định; collector no-op không tạo file.
- [x] Public API `__all__` không đổi; `import` OK.
- [x] Failure paths log rõ, không crash thô; exception công khai giữ nguyên.

## Risk Assessment

- **Rủi ro:** sửa vòng solve gây regression im lặng. **Giảm thiểu:** test-first lock hành vi; chạy integration test opt-in (`pytest -m integration`) nếu môi trường cho phép.
- **Rủi ro:** field config mới phá `__post_init__`. **Giảm thiểu:** giữ nguyên sentinel pattern, chỉ append field.
- **Rủi ro:** wiring collector vào 2 solver lệch nhau. **Giảm thiểu:** checklist "sửa cả hai" + test gọi cả sync/async khởi tạo collector.
