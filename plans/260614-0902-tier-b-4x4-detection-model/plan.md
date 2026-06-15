---
title: 'Tier B: custom 4x4 detection model (bbox pipeline + train + integrate)'
description: ''
status: completed
priority: P2
branch: feat/data-flywheel
tags: []
blockedBy: []
blocks: []
created: '2026-06-14T02:06:35.889Z'
createdBy: 'ck:plan'
source: skill
---

# Tier B: custom 4x4 detection model (bbox pipeline + train + integrate)

## Overview

Tầng B của data flywheel: train **custom detection model** cho 4x4 — đúng bản chất "chọn ô có X" (1 object lớn trải nhiều ô) cho 7 lớp COCO thiếu (bridges, chimneys, crosswalks, mountains, palm trees, stairs, tractors). Thay per-cell classification fallback (cầu nối) bằng detection thật khi model sẵn sàng.

**Spec gốc:** `plans/reports/brainstorm-260614-0805-solver-fast-high-success-fix-report.md` (§Tầng B). Quyết định 14/06: detection model + pipeline bbox MỚI (không tái dùng được data classification của flywheel).

**Mode:** `--tdd` cho code (collector/dataset/integrate); train là GPU thủ công, test khô.

**Bối cảnh data (cốt lõi):** flywheel hiện thu **per-cell tile (classification)**. Detection cần **ảnh 4x4 full + bbox**. → Phase 1–2 dựng pipeline data MỚI; KHÔNG tái dùng `collected/` per-cell hiện có cho detection.

**Bất biến:**
- Public API `__all__` không đổi; collector classification cũ (per-cell) GIỮ NGUYÊN, chỉ THÊM nhánh full-image.
- `training/` không vào wheel; dataset/ảnh raw không commit.
- Detection model mới load song song COCO; verify SHA256 như model classification.
- `solver.py` + `async_solver.py` đối xứng nếu đụng solve path.

## Phases

| Phase | Name | Status |
|-------|------|--------|
| 1 | [Full-image 4x4 collection](./phase-01-full-image-4x4-collection.md) | Completed |
| 2 | [Cell-bbox annotation + detection dataset](./phase-02-cell-bbox-annotation-detection-dataset.md) | Completed |
| 3 | [Detection training + export + SHA256](./phase-03-detection-training-export-sha256.md) | Completed |
| 4 | [Integrate custom 4x4 detection model + publish](./phase-04-integrate-custom-4x4-detection-model-publish.md) | Completed |

## Implementation Log (cook --tdd — 2026-06-14, all 4 phases)

### Update (autonomous run): Phase 3 + collect driver done
- **Phase 3:** `train_detection.py` (YOLO detect, device auto CUDA>MPS>CPU, `--amp/--no-amp`, `--resume`), `write_model_card.py`, reuse `export_onnx.py`/`compute_sha256.py`; `device_utils.resolve_device`. `collect.py` (loop driver + progress counters). Tests: `test_train_detection_args.py`.
- **Smoke (verified end-to-end on MPS):** synthetic YOLO-detect dataset → `train_detection.train(device=mps, base=yolo11n.pt, 1 epoch)` → produced `best.pt` (args.yaml: task=detect, device=mps). Export needs `onnx` pkg → added to dev extras (`onnx`, `onnxslim`); runtime keeps `onnxruntime` only.
- **Gate (final):** 145 unit passed, 1 deselected; ruff `src/`+`training/` clean; mypy `src/` 4 pre-existing only.
- **STILL BLOCKED on human:** a real custom model needs collected 4x4 images **annotated by a person** (`annotate_detection_cli`). Cannot be automated (no auto pseudo-label). Pipeline is code-complete + smoke-proven; the trained artifact awaits annotation.

### Original (Phases 1/2/4)

- **Phase 1:** `DataCollector.record_challenge_image` (ảnh 4x4 full → `collected/full/`, metadata riêng); hook ở `square_handler`; `_append_metadata(subdir=)`. Tests: `test_data_collector.py` (+3).
- **Phase 2:** `class_mapping.DETECTION_CLASSES`/`detection_class_id`; `prepare_detection_dataset.py` (`cell_to_yolo_bbox` + YOLO detect dataset + data.yaml); `annotate_detection_cli.py`. Tests: `test_detection_dataset.py`.
- **Phase 4:** `types.CUSTOM_DETECTION_CLASSES`/`CUSTOM_DETECTION_TARGET_MAPPINGS`; `SolverConfig.custom_detection_model_path`; `YOLODetector` custom detect (load+SHA256 verify+`has_custom_detection`+`get_custom_detection_class`+`detect_for_grid_custom`); `square_handler` 3-tier (COCO→custom→per-cell); wiring 2 solver đối xứng. Tests: `test_custom_detection_integration.py`. Docs: `training-and-flywheel.md` mục Tier B.
- **Phase 3 (DEFERRED):** train_detection.py + train thật cần GPU + data đã annotate — chưa làm theo yêu cầu.
- **Gate:** 138 unit passed, 1 deselected; ruff `src/`+`training/` clean; mypy `src/` chỉ 4 lỗi pre-existing. Public API `__all__` không đổi; mặc định (no custom model) = hành vi y hệt plan 0805.
- **Code review:** DONE (7/7 acceptance, 0 Critical/High/Medium). SHA256 constant trống tới Phase 3 (chủ ý).

## Build order

1. **Phase 1** — thu ảnh 4x4 full + metadata (nền data detection).
2. **Phase 2** — annotate cell→bbox + builder dataset YOLO detect (data.yaml).
3. **Phase 3** — train_detection.py (GPU thủ công) + export ONNX + SHA256 + model_card.
4. **Phase 4** — tích hợp model detection custom vào path 4x4 (ưu tiên hơn per-cell fallback) + config + publish HF.

## Dependencies

- **blockedBy:** `260614-0805-solver-fast-high-success-fix` (đã completed) — per-cell fallback là cầu nối; Tầng B thay thế khi model sẵn sàng.
- Dùng tooling `training/` của `260613-1719-recaptcha-suite-data-flywheel` (completed) làm khuôn (export/sha256/class_mapping pattern).
- **Train (Phase 3):** chạy trên **Mac M2 Max MPS** (`--device mps`, dataset Tier B nhỏ, data local) HOẶC cloud CUDA (Colab) nếu dataset lớn. Scripts auto-detect device (`device_utils.resolve_device`). Phase 1,2,4 codeable + test khô local.

## Kỳ vọng thực tế (brutal honesty)

- Đây là plan DÀI + nặng data (thu + annotate bbox thủ công nhiều mẫu mới có model tốt). Không có shortcut.
- Cell-level bbox (mỗi ô được chọn = 1 box) là weak-supervision — nhanh để annotate (tái dùng grid) nhưng box thô hơn bbox chặt; chấp nhận cho vòng đầu, tinh chỉnh sau.
- Kết quả train thật phụ thuộc lượng data thu được; plan giao pipeline + script + flow, KHÔNG cam kết accuracy cụ thể.
