---
phase: 2
title: "Data Collection Pipeline (sync+async)"
status: completed
priority: P1
effort: "2-3d"
dependencies: [1]
---

# Phase 2: Data Collection Pipeline (sync+async)

<!-- Updated: Validation Session 1 - hook thu thập tile chuyển từ base_handler sang YOLODetector (tái dùng crop sẵn có, DRY) -->

## Overview

Biến collector no-op thành pipeline ghi thật: khi `collect_data=True`, lưu tile **uncertain** (conf giữa `min_confidence_threshold` và `conf_threshold`), **failed** (solve fail / `TokenExtractionError`), và **unknown keyword** (không có trong mapping) vào `collect_dir` kèm metadata JSONL. Hoạt động cho cả `RecaptchaSolver` và `AsyncRecaptchaSolver`.

## Requirements

- Functional:
  - Bật `collect_data=True` → solve sinh file PNG tile + dòng metadata.
  - Layout: `collected/{YYYY-MM-DD}/{captcha_type}/{pred_class}_{conf:.2f}_{uuid8}.png`.
  - Metadata: `collected/metadata.jsonl`, mỗi dòng JSON: `{ts, captcha_type, keyword, predicted_class, confidence, reason, image_path, solve_outcome}`.
  - 3 lý do thu (`reason`): `uncertain` | `failed` | `unknown_keyword`.
  - 3x3: **tái dùng tile crop sẵn có trong `YOLODetector.classify_tiles_with_confidence`** (yolo_detector.py:518-527 đã crop tile). KHÔNG crop lại ở handler (DRY).
  - Async: ghi đĩa chạy trong thread pool (không block event loop), tái dùng pattern offload hiện có (`_run_in_executor`).
- Non-functional:
  - Tắt (`collect_data=False`) → zero I/O, zero overhead (đã đảm bảo Phase 1).
  - Ghi an toàn đồng thời nhiều solver (mỗi solver `collect_dir` riêng, hoặc append JSONL an toàn qua lock).
  - Không làm chậm solve đáng kể khi bật (ghi nền, không chặn click/verify).

## Architecture

- `DataCollector.record_tile(image, cell, confidence, captcha_type, keyword)`: **collector tự áp ngưỡng** (`min_confidence_threshold ≤ conf < conf_threshold` → reason=`uncertain`), tạo thư mục ngày/captcha_type, ghi PNG (`cv2.imwrite`), append 1 dòng JSONL dưới `threading.Lock`. Tile dưới ngưỡng/đủ chắc → bỏ qua.
- `DataCollector.record_failure(...)`: ghi metadata cho lần solve fail (kèm ảnh nếu có).
- **Hook points (quyết định Validation: hook ở TẦNG DETECTOR, không ở handler):**
  - `YOLODetector` nhận `collector` optional (inject từ solver). Trong `classify_tiles_with_confidence`, sau khi đã có sẵn `tiles[i]` + `confidences[i]`, forward `(tile, cell, conf)` cho `collector.record_tile(...)` khi collector bật. Tái dùng crop sẵn có → DRY, không lệch toạ độ.
  - Threshold-uncertain do collector quyết (collector giữ tham chiếu config/thresholds).
  - `solver._get_target_class` (và async tương đương): keyword không map được → `collector.record_failure(reason="unknown_keyword")` + lưu ảnh challenge nếu có.
  - Vòng solve khi đi tới `TokenExtractionError` → `record_failure(reason="failed")`.
- Async: `AsyncRecaptchaSolver` bọc `collector.record_*` bằng `_run_in_executor` (giữ collector cùng instance, lock thread-safe).

## Related Code Files

- Modify: `src/vision_ai_recaptcha_solver/collection/collector.py` (ghi thật + áp ngưỡng uncertain)
- Modify: `src/vision_ai_recaptcha_solver/detector/yolo_detector.py` (nhận `collector` optional; forward `(tile, cell, conf)` trong `classify_tiles_with_confidence`)
- Modify: `src/vision_ai_recaptcha_solver/solver.py`, `async_solver.py` (inject collector vào detector; record_failure cho failed/unknown_keyword; async offload)
- Create: `tests/test_data_collector.py`
- Note: handlers (`dynamic/selection/square`) **không cần sửa** cho uncertain (detector lo) — chỉ đụng nếu cần forward thêm context.

## Implementation Steps (TDD)

1. **Test trước:**
   - `tests/test_data_collector.py`: với `collect_data=True` + `collect_dir=tmp_path`, gọi `record_tile` với conf uncertain → 1 PNG + 1 dòng JSONL đúng schema. Gọi `record_failure` → metadata đúng. `collect_data=False` → không file.
   - Test crop cell 3x3: ảnh 300x300 → 9 crop 100x100 đúng index.
   - Test async: gọi từ event loop không raise; file được ghi.
2. Hiện thực ghi PNG + JSONL (lock) + áp ngưỡng uncertain trong `DataCollector`.
3. `YOLODetector` nhận `collector` optional; trong `classify_tiles_with_confidence` forward `(tile, cell, conf)` cho collector khi bật (tái dùng tile đã crop, không crop lại).
4. Inject collector từ `solver.py`/`async_solver.py` vào detector khi khởi tạo.
5. Cắm hook `unknown_keyword` + `failed` vào `solver.py` và `async_solver.py` (đối xứng).
6. Async offload ghi đĩa qua `_run_in_executor`.
7. ruff + mypy + pytest. Verify thủ công 1 lần với `collect_data=True` (nếu chạy được solve thật/integration).

## Success Criteria

- [x] Test thu thập viết TRƯỚC và xanh (sync + async + crop).
- [x] Bật collect → PNG + JSONL đúng layout/schema cho cả 3 reason.
- [x] Tắt collect → zero I/O (test khẳng định).
- [x] Cả `RecaptchaSolver` và `AsyncRecaptchaSolver` đều thu được.
- [x] ruff + mypy + pytest xanh; không regression.

## Risk Assessment

- **Rủi ro:** ghi đĩa làm chậm/àm event loop async. **Giảm thiểu:** offload executor; benchmark nhẹ.
- **Rủi ro:** crop cell sai toạ độ → ảnh rác. **Giảm thiểu:** test crop với ảnh kích thước cố định.
- **Rủi ro:** đua ghi JSONL khi nhiều solver. **Giảm thiểu:** lock + khuyến nghị `collect_dir` riêng mỗi solver.
- **Rủi ro:** lệch sync/async (chỉ 1 bên thu). **Giảm thiểu:** test cả hai; checklist sửa cả hai.
