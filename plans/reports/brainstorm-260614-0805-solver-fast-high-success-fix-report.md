# Brainstorm — Solver: fix errors + faster + higher success

Date: 2026-06-14 | Status: APPROVED (Tầng A + per-cell fallback; Tầng B deferred)
Branch: feat/data-flywheel

## Problem statement

Integration test (`tests/integration/test_google_demo.py`) fail sau 9m38s với `TokenExtractionError` — solver đốt hết 12 attempts không giải được. Mục tiêu: sửa lỗi, chạy nhanh, success cao. Không phải regression của flywheel (collector mặc định tắt; đi đúng path cũ).

## Root cause (scout, verified)

- **4x4 COCO gap (chính):** 4x4 dùng `SquareCaptchaHandler` → `detect_for_grid` → COCO `yolo12x.pt`. `COCO_TARGET_MAPPINGS` chỉ phủ **8/15** lớp (bicycle, car/taxi, motorcycle, bus, boat, traffic light, fire hydrant, parking meter). Thiếu 7: bridges, chimneys, crosswalks, mountains, palm trees, stairs, tractors → 4x4 các lớp này không giải được (`get_coco_target_class`→None → `[]` → reload).
- **Model 57k (14 lớp, đủ stairs/bridges...) chỉ dùng 3x3**, bị bỏ qua ở 4x4.
- **Chậm:** 12 attempts, mỗi attempt download+infer+nhiều `human_delay`+`wait_for_verify_result`/reload @ `default_timeout=10s`. Khi liên tục gặp 4x4-unsupported → đốt gần full timeout mỗi lượt thay vì bỏ nhanh.
- **Không fail-fast / không ưu tiên challenge giải được.**

## Brutal honesty

"Fix ALL + 100% success" trên reCAPTCHA thật = không khả thi tuyệt đối (adversarial, xác suất). 4x4 cho 7 lớp thiếu chỉ giải đúng bản chất khi có detection model train trên các lớp đó (cần GPU + data). Near-term: per-cell classification fallback (dùng 57k) làm cầu nối + fail-fast + speed → success cao thực tế + nhanh.

## Approaches đã cân nhắc

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| Fail-fast skip unsupported | rẻ, nhanh, no GPU, nhiều reload hơn → trúng challenge giải được | phụ thuộc Google serve | ✅ (A1) |
| Per-cell classification fallback 4x4 (57k) | phủ ngay 14 lớp cho 4x4, no GPU | object lớn trải nhiều cell có thể kém hơn detection | ✅ (chọn thêm) |
| Train detection model 7 lớp | đúng bản chất 4x4 | cần GPU + data flywheel, lâu | ✅ deferred (B1) |
| Giảm human_delay mạnh | nhanh nhất | tăng rủi ro anti-bot | ❌ (chỉ balanced) |

## Giải pháp chốt

### Tầng A — implement + verify ngay (no GPU)

**A1. Fail-fast skip challenge không giải được (sync + async đối xứng)**
- `YOLODetector.is_supported(keyword, captcha_type)`: 4x4→COCO map; 3x3→classification map (sau khi per-cell fallback vào, 4x4 cũng coi như supported nếu classification có lớp → xem A4).
- Solve loop: unsupported → reload ngay, delay tối thiểu, "skip" rẻ (không đốt full `default_timeout`). Ngân sách wall-clock qua `timeout`. Nhiều reload-skip hơn → xác suất trúng challenge giải được tăng.

**A2. Speed (balanced)**
- Bỏ `human_delay`/wait thừa trên path reload/skip; **giữ** human_delay trên path click tile + verify (anti-bot). Reload timeout ngắn hơn.

**A3. Integration test = retry-until-solvable**
- Test tự retry `solve` vài lần (bounded vòng + wall-clock) tới khi trúng challenge giải được; **vẫn assert token** cuối cùng. Hết flaky, không hạ chuẩn assert. (Thay quyết định strict-một-lần của plan 260418 bằng strict-có-retry — user đã duyệt.)

**A4. Per-cell classification fallback cho 4x4 (NEW, user chọn thêm)**
- `SquareCaptchaHandler`: thử COCO detection trước (nếu `coco_class` có); nếu COCO không có lớp → fallback **chia 4x4 thành 16 cell, classify từng cell bằng model 57k** (dùng classification `target_class` mà solver đã truyền vào `handler.solve`), chọn cell có conf ≥ `conf_threshold` (tái dùng `classify_tiles_with_confidence` với grid_size=4 — đã có sẵn, DRY). Phủ 14 lớp cho 4x4 ngay.
- Collector flywheel tự động thu tile 4x4 uncertain qua hook detector sẵn có → data cho B1.

### Tầng B — deferred (cần GPU)
**B1. Train detection model** 7 lớp thiếu qua flywheel đã scaffold (collect→review→prepare→train→export→SHA256→publish). Là đường dài đúng bản chất; per-cell fallback (A4) là cầu nối tới khi xong.

## Acceptance / success metrics

- Unit suite vẫn 107 passed (no regression); thêm test cho `is_supported` + per-cell 4x4 fallback (mock detector).
- Solve nhanh hơn rõ rệt khi gặp unsupported (fail-fast, không treo ~10 phút).
- 4x4 phủ 14 lớp (per-cell fallback) — verify bằng unit test mock; success thật phụ thuộc chất lượng classify per-cell.
- Integration test xanh ổn định (retry-until-solvable, assert token).
- ruff/mypy `src/` clean; public API `__all__` không đổi; cả sync+async sửa đối xứng.

## Risks & mitigation

- **Per-cell 4x4 kém với object lớn trải nhiều cell** → chấp nhận như cầu nối; B1 (train) là fix thật; collector thu data để cải thiện.
- **Fail-fast reload quá nhiều → reCAPTCHA "try again later"** → giữ giới hạn reload + human_delay path click; cap wall-clock.
- **Sync/async lệch** → checklist sửa cả hai + test cả hai.
- **Retry-until-solvable vẫn có thể chậm/khó nếu Google serve toàn unsupported** → bounded retry + timeout; coi skip-graceful trong giới hạn là hợp lệ trước khi assert token (vẫn ưu tiên token).

## Next steps

- `/ck:plan --tdd` (modifies critical solve-loop business logic + có test coverage cần bảo toàn → tests-first).
- Implement Tầng A (A1–A4) + verify (unit + integration retry).
- B1 train: khi có GPU + đủ data review.

## Unresolved questions

1. Ngân sách wall-clock cho integration retry-until-solvable bao nhiêu (vd 2–3 phút)? — chốt khi plan.
2. Per-cell 4x4: ngưỡng conf riêng hay tái dùng `conf_threshold`? — đề xuất tái dùng, tinh chỉnh sau theo data.
