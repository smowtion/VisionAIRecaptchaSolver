---
title: "reCAPTCHA Suite — monorepo + data flywheel"
description: ""
status: completed
priority: P2
branch: "main"
tags: []
blockedBy: []
blocks: []
created: "2026-06-13T10:30:06.449Z"
createdBy: "ck:plan"
source: skill
---

# reCAPTCHA Suite — monorepo + data flywheel

## Overview

Hợp nhất `VisionAIRecaptchaSolver` (runtime) + `recaptcha-classification-57k` (model/dataset) thành 1 monorepo có vòng lặp dữ liệu (active learning): solve → collect tile uncertain/failed → review thủ công → train cloud GPU → export ONNX + SHA256 → solver auto-download model mới.

**Spec gốc:** `plans/reports/brainstorm-260613-1719-recaptcha-suite-data-flywheel-report.md` (đã duyệt).

**Mode:** `--tdd` — mỗi phase code viết test trước (lock hành vi solver hiện tại trước khi thêm tính năng).

**Nguyên tắc bất biến:**
- Public API trong `__init__.py __all__` giữ nguyên (back-compat).
- `solver.py` + `async_solver.py` là 2 impl song song → **mọi thay đổi logic phải làm cả hai**.
- Collector **opt-in, mặc định TẮT** (`collect_data=False`) — user PyPI không bị ảnh hưởng.
- `training/` **không** vào wheel (kiểm `tool.setuptools.packages.find`).
- Dataset raw + `collected/` **không** commit (gitignore + HF/git-lfs).
- reCAPTCHA chỉ trả pass/fail → không auto-label; nhãn từ human review.

## Phases

| Phase | Name | Status |
|-------|------|--------|
| 1 | [Setup & Stabilize Solver + Collector Scaffold](./phase-01-setup-stabilize-solver-collector-scaffold.md) | Done |
| 2 | [Data Collection Pipeline (sync+async)](./phase-02-data-collection-pipeline-sync-async.md) | Done |
| 3 | [Repo Integration (dataset/training/class-mapping/review)](./phase-03-repo-integration-dataset-training-class-mapping-review.md) | Done |
| 4 | [Training Loop (cloud GPU + export + SHA256 + HF)](./phase-04-training-loop-cloud-gpu-export-sha256-hf.md) | Done |

## Implementation Log (cook — 2026-06-13)

- **Phase 1:** `SolverConfig.collect_data`/`collect_dir` (opt-in, default off); `collection/DataCollector` (no-op when disabled); wired into `solver.py` + `async_solver.py` + `YOLODetector(collector=...)`; `.gitignore` `collected/`. Tests: `test_config.py` (+7), `test_collector_scaffold.py`.
- **Phase 2:** Collector writes PNG + `metadata.jsonl` (reasons `uncertain`/`failed`/`unknown_keyword`); tile hook in `YOLODetector.classify_tiles_with_confidence` (reuses cropped tiles, DRY); failure hooks in both solvers (`_get_target_class(browser, captcha_type)`); async offload via `_run_in_executor`. Tests: `test_data_collector.py`.
- **Phase 3:** `training/class_mapping.py` (folder↔class_id↔label, 14 classes, validated vs `types.CLASS_NAMES`), `prepare_dataset.py`, `review_cli.py`; moved `train_model/` → `training/`; `pyproject` pytest pythonpath +`training`. Tests: `test_class_mapping.py`, `test_prepare_dataset.py`.
- **Phase 4:** `training/train.py` (from notebook, CLI + `--resume`), `export_onnx.py` (`--weights`), `compute_sha256.py`; `docs/training-and-flywheel.md`. Tests: `test_training_scripts_args.py` (dry, no GPU).
- **Gate:** 107 passed, 1 deselected (integration); ruff `src/`+`training/` clean; mypy `src/` only 4 pre-existing errors (image_utils, yolo_detector — present on HEAD). Public API `__all__` unchanged.
- **Code review:** DONE_WITH_CONCERNS → fixed C1 (collector `except Exception`, never aborts solve), M1 (docstring), H1 (notebook lint clean).
- **Not run locally (env lacks pip/setuptools/build):** `python -m build` wheel-exclusion check — guaranteed by config (`packages.find where=src`, `training/` outside `src/`, no `__init__.py`). Real cloud-GPU train run (Phase 4) — out of machine scope by design.

## Dependencies

- Không bị block bởi plan nào. Plan `260418-1538-google-demo-integration-test` đã `completed`; integration test ở đó dùng để verify regression ở Phase 1.
- Phase 2 phụ thuộc collector scaffold của Phase 1. Phase 3 phụ thuộc class-name chuẩn hóa (dùng ở Phase 2 metadata). Phase 4 phụ thuộc dataset gộp của Phase 3.

## Build order & ưu tiên

1. **Phase 1** (P1) — nền tảng: cài đặt ổn định + fail graceful + collector scaffold no-op.
2. **Phase 2** (P1) — collect uncertain/failed/unknown (sync+async).
3. **Phase 3** (P2) — gộp repo + dataset + class_mapping + review CLI.
4. **Phase 4** (P2) — training loop cloud GPU + export + SHA256 + publish HF.

## Validation Log

### Verification Results (Session 1)
- Tier: Standard (4 phases). Claims checked: ~10.
- Verified: 10 | Failed: 0 | Unverified: 0.
- Đã xác minh tồn tại: `config.py` sentinel `_UNSET` + thresholds; `base_handler.__init__`; `solver.py` solve loop + `_get_target_class` + `_get_handler`; `async_solver.py` `ThreadPoolExecutor`/`_run_in_executor`; `yolo_detector.py` `MODEL_SHA256` (line 43), `_download_model` (245), `classify_tiles_with_confidence` crop tile nội bộ (518-527, trả `list[(cell,conf)]`); `grid_utils.py`/`image_utils.py` helpers; `export_onnx.py` hardcode weights path.
- **Phát hiện thiết kế:** detector đã crop tile sẵn nhưng không lộ ra → hook thu thập nên ở tầng detector (DRY), không re-crop ở handler. → đổi Phase 1+2.

### Quyết định phỏng vấn (Session 1)
1. **Hook thu thập tile:** trong `YOLODetector.classify_tiles_with_confidence` (tái dùng crop sẵn, DRY) — KHÔNG ở base_handler. → cập nhật Phase 1 (wiring vào detector) + Phase 2 (forward tile từ detector, handlers không đổi).
2. **Review tool:** CLI thuần (click), xem ảnh bằng trình xem ngoài. Không dựng HTML vòng này. → Phase 3 giữ nguyên.
3. **Model versioning:** filename có version + `model_card.json` sidecar (date/epochs/imgsz/classes/sha256/dataset_size). → Phase 4 chốt.
4. **Publish HF:** thủ công `huggingface-cli` + checklist docs; không tự động hóa. → Phase 4 chốt.

### Whole-Plan Consistency Sweep (Session 1)
- Rà toàn bộ `plan.md` + 4 phase: thuật ngữ "hook ở base_handler"/"crop ở handler" đã được thay bằng "hook ở detector" ở Phase 1 + Phase 2 (Related Files, Architecture, Steps). Không còn tham chiếu mâu thuẫn.
- Phase 4 unresolved Q2/Q3 (versioning, HF publish) đã chốt; không còn "(quyết định khi cook)".
- 0 mâu thuẫn còn lại. Plan đủ điều kiện implement (Failed: 0).
