---
phase: 3
title: Integration retry-until-solvable & verify
status: completed
priority: P2
effort: 0.5d + thời gian chạy integration
dependencies:
  - 2
---

# Phase 3: Integration retry-until-solvable & verify

## Overview

Sửa `tests/integration/test_google_demo.py` thành **retry-until-solvable**: gọi `solve` lặp lại (bounded vòng + wall-clock) cho tới khi trúng challenge giải được; **vẫn assert token** cuối cùng (không hạ chuẩn). Sau đó verify toàn bộ: unit suite + ruff + mypy, rồi chạy integration thật.

## Requirements

- Functional:
  - Test: retry `solver.solve(...)` tối đa `N` lần (đề xuất N=3) hoặc tới wall-clock budget (đề xuất ~3 phút), bắt `TokenExtractionError` giữa các lần; thành công sớm khi có token; assert `result.token` cuối cùng. Mỗi lần solve dùng config timeout riêng (ngắn hơn nhờ fail-fast Phase 1).
  - Giữ marker `@pytest.mark.integration` + headless; không vào CI mặc định.
- Non-functional:
  - Không hạ chuẩn: vẫn phải ra token để pass (retry chỉ chống non-determinism, không bỏ assert).
  - Bounded: không treo vô hạn.

## Architecture

- `tests/integration/test_google_demo.py`: vòng retry quanh `solve`, dùng `time.monotonic()` để cap wall-clock; log mỗi lần thử (captcha_type/outcome). Có thể giảm `max_attempts`/`timeout` trong config test để mỗi solve nhanh.
- Wall-clock budget (unresolved Q1 report — chốt): N=3 retry, cap ~180s tổng.

## Related Code Files

- Modify: `tests/integration/test_google_demo.py` (retry-until-solvable)
- Read for context: `src/vision_ai_recaptcha_solver/solver.py` (timeout/max_attempts)

## Implementation Steps (TDD)

1. Sửa test integration thành retry-until-solvable (bounded N + wall-clock), giữ assert token.
2. **Verify khô (no browser):** full `pytest` (unit) xanh — gồm Phase 1+2 tests; ruff `src/`+`training/` clean; mypy `src/` không lỗi mới.
3. **Verify thật:** chạy `pytest -m integration` (Chrome + network). Ghi lại kết quả + thời gian; nếu fail do Google serve toàn unsupported trong budget → ghi nhận honest (limitation), không coi là regression.
4. Cập nhật `docs/` nếu cần (ghi chú fail-fast + per-cell 4x4 fallback trong `codebase-summary.md`).

## Success Criteria

- [ ] Integration test retry-until-solvable, vẫn assert token, bounded.
- [ ] full `pytest` (unit) xanh; ruff + mypy `src/` clean.
- [ ] Integration chạy thật: nhanh hơn rõ rệt so với 9m38s baseline (fail-fast); pass khi trúng challenge giải được trong budget.
- [ ] Không regression public API/exception.

## Risk Assessment

- **Google serve toàn unsupported trong budget → vẫn fail:** bounded retry + honest reporting; per-cell 4x4 (Phase 2) đã giảm mạnh khả năng này (4x4 giờ phủ 14 lớp).
- **Integration vẫn flaky bản chất (live reCAPTCHA):** chấp nhận; marker opt-in, không vào CI mặc định; retry giảm flakiness.
- **Thời gian chạy integration dài:** cap wall-clock; chạy thủ công, không bắt buộc CI.
