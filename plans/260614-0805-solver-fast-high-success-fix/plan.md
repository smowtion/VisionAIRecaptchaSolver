---
title: 'Solver: fail-fast + per-cell 4x4 fallback + faster, higher success'
description: ''
status: completed
priority: P2
branch: feat/data-flywheel
tags: []
blockedBy: []
blocks: []
created: '2026-06-14T01:30:45.076Z'
createdBy: 'ck:plan'
source: skill
---

# Solver: fail-fast + per-cell 4x4 fallback + faster, higher success

## Overview

Sửa solver: chạy nhanh hơn (fail-fast bỏ challenge không giải được) + success cao hơn (per-cell classification fallback phủ 14 lớp cho 4x4) + integration test xanh ổn định (retry-until-solvable, vẫn assert token).

**Spec gốc:** `plans/reports/brainstorm-260614-0805-solver-fast-high-success-fix-report.md` (đã duyệt: Tầng A + per-cell fallback; Tầng B train deferred).

**Mode:** `--tdd` — sửa solve-loop business logic + có test coverage cần bảo toàn → tests-first mỗi phase.

**Bất biến:**
- Public API `__all__` không đổi; exception công khai (`TokenExtractionError`...) không đổi.
- `solver.py` + `async_solver.py` song song → mọi thay đổi logic làm CẢ HAI.
- Giữ `human_delay` trên path click tile + verify (anti-bot); chỉ trim trên path reload/skip.
- Không hạ chuẩn assert token của integration test (retry, không bỏ assert).
- Collector flywheel: per-cell 4x4 fallback dùng `classify_tiles_with_confidence` → hook collector tự thu tile 4x4 (bonus data cho Tầng B).

## Phases

| Phase | Name | Status |
|-------|------|--------|
| 1 | [Fail-fast & speed (is_supported + skip)](./phase-01-fail-fast-speed-is-supported-skip.md) | Completed |
| 2 | [Per-cell 4x4 classification fallback](./phase-02-per-cell-4x4-classification-fallback.md) | Completed |
| 3 | [Integration retry-until-solvable & verify](./phase-03-integration-retry-until-solvable-verify.md) | Completed |

## Implementation Log (cook --tdd — 2026-06-14)

- **Phase 1:** `YOLODetector.is_supported`; solve loop (sync+async) tách `attempts`/`skips`, fast-skip unsupported (`_reload_challenge` delay tối thiểu), `solved` flag + short token-wait khi fail (không treo full timeout); xoá `_get_target_class`. Tests: `test_is_supported.py`.
- **Phase 2:** `SquareCaptchaHandler` per-cell classification fallback (`classify_tiles_with_confidence` grid_cells=4) khi COCO không có lớp; `is_supported` 4x4 = COCO OR classification. Tests: `test_square_handler_fallback.py`.
- **Phase 3:** integration test retry-until-solvable (bounded N=3 + wall-clock 180s, vẫn assert token).
- **Gate:** 117 unit passed, 1 deselected; ruff `src/` clean; mypy `src/` chỉ 4 lỗi pre-existing (image_utils, yolo_detector — có trên HEAD). Public API/exception không đổi. sync+async đối xứng.
- **Code review:** DONE_WITH_CONCERNS (6/6 acceptance, 0 Critical/High/Medium) → áp 2 fix Low: off-by-one `skips < max_skips`, test sentinel `-1`.
- **Chưa verify thật:** integration live (Chrome+network) chạy khi user muốn; Tầng B train (GPU) deferred.

## Build order

1. **Phase 1** — `is_supported` + fail-fast skip + speed (nền tảng; 4x4 supported = COCO).
2. **Phase 2** — per-cell 4x4 fallback; mở rộng `is_supported` (4x4 supported = COCO HOẶC classification).
3. **Phase 3** — integration retry-until-solvable + verify toàn bộ.

## Dependencies

- Không bị block. Tầng B (train detection model 7 lớp thiếu) dùng tooling `training/` của plan `260613-1719-recaptcha-suite-data-flywheel` (đã `completed`) — deferred, cần GPU, ngoài plan này.
- Per-cell fallback (Phase 2) là cầu nối tới khi Tầng B train xong.

## Kỳ vọng thực tế (brutal honesty)

- 3x3: success cao (model 57k đủ 14 lớp). 4x4: per-cell fallback phủ 14 lớp nhưng có thể kém với object lớn trải nhiều cell → Tầng B (train) là fix thật.
- "100% success mọi challenge" KHÔNG phải deliverable của plan này; đạt được tối đa-thực-tế + nhanh + test xanh.
