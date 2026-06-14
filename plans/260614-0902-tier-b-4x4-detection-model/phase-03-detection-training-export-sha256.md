---
phase: 3
title: Detection training + export + SHA256
status: completed
priority: P2
effort: 0.5d code + train time GPU ngoài máy
dependencies:
  - 2
---

# Phase 3: Detection training + export + SHA256

## Overview

Script train YOLO **detection** trên dataset Phase 2, export ONNX, tính SHA256, sinh model_card. Test khô (parse args, không train). Train thật: **Mac M2 Max MPS** (`--device mps`, dataset nhỏ, data local) hoặc cloud CUDA (Colab) nếu lớn — `train_detection.py` tái dùng `device_utils.resolve_device` (auto CUDA>MPS>CPU) + `--amp/--no-amp` (như `train.py` đã làm).

## Requirements

- Functional:
  - `training/train_detection.py`: `train(data="...data.yaml", epochs, imgsz=640, batch, device=None, amp=True, ...)` dùng `YOLO("yolo11x.pt")` (detect, không phải -cls); CLI click; `--resume`; `--device auto` (qua `device_utils.resolve_device`) + `--amp/--no-amp`. Import lười `ultralytics` (import/`--help` chạy không GPU).
  - Export: tái dùng `training/export_onnx.py` (đã tham số `--weights`) — detect weights export ONNX dynamic.
  - SHA256: tái dùng `training/compute_sha256.py`.
  - model_card sidecar: `{date, task:"detect", classes:DETECTION_CLASSES, epochs, imgsz, sha256, dataset_size}` (script nhỏ hoặc flag).
- Non-functional:
  - Không cần GPU để import/test khô. Giữ verify SHA256 làm cổng an toàn.

## Architecture

- `train_detection.py`: gần `train.py` (classification) nhưng `data=data.yaml`, base `yolo11x.pt`, task detect. Output `runs/detect/<name>/weights/best.pt`.
- Tái dùng export_onnx + compute_sha256 (không lặp).
- model_card: thêm `training/write_model_card.py` hoặc flag trong export.

## Related Code Files

- Create: `training/train_detection.py`
- Reuse: `training/export_onnx.py`, `training/compute_sha256.py`
- Create: `training/write_model_card.py` (hoặc flag)
- Create: `tests/test_train_detection_args.py` (khô: import, parse args, --help)

## Implementation Steps (TDD)

1. **Test trước (khô, no GPU):** `test_train_detection_args.py` — import `train_detection` không lỗi; `train` defaults đúng (imgsz=640...); `main --help` exit 0; (nếu viết write_model_card: card chứa task=detect + sha256 hợp lệ cho file tmp).
2. Viết `train_detection.py` (lười import ultralytics).
3. Viết `write_model_card.py` (hoặc flag export).
4. Doc flow train detect cloud GPU (kéo dataset → train_detection → export → sha256 → model_card).
5. ruff + mypy + pytest (khô).
6. (Ngoài máy) train thật 1 vòng ngắn GPU xác nhận script; ghi kết quả; KHÔNG bắt buộc CI.

## Success Criteria

- [ ] Test khô args/model_card viết TRƯỚC và xanh (no GPU).
- [ ] `train_detection.py --help` chạy; export nhận detect weights.
- [ ] compute_sha256 khớp giá trị dán vào detector; model_card đầy đủ.
- [ ] Doc flow detect train→export→card đầy đủ.
- [ ] ruff + mypy + pytest clean.

## Risk Assessment

- **Không GPU verify train trong CI/local:** test khô + smoke; train thật thủ công, ghi lại.
- **Đổi số/thứ tự class detect → solver mapping sai:** cố định DETECTION_CLASSES; verify SHA256; doc cảnh báo.
- **Dataset quá nhỏ → model yếu:** ghi nhận honest; per-cell fallback vẫn là backstop tới khi đủ data.
