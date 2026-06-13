---
phase: 4
title: "Training Loop (cloud GPU + export + SHA256 + HF)"
status: completed
priority: P2
effort: "1-2d (code) + train time ngoài máy"
dependencies: [3]
---

# Phase 4: Training Loop (cloud GPU + export + SHA256 + HF)

# Khép vòng flywheel: train lại model trên dataset đã gộp (cloud GPU thủ công), export ONNX, tính SHA256, cập nhật `YOLODetector.MODEL_SHA256`, và tài liệu hóa flow publish lên Hugging Face để solver auto-download model mới.

## Overview

Chuẩn hóa script train + export + verify + publish. **Train chạy ngoài máy (cloud GPU)** vì Mac không CUDA; repo chỉ cung cấp script + tài liệu + bước verify SHA256 + cập nhật model reference. Không tự động hóa retrain theo lịch (ngoài scope).

## Requirements

- Functional:
  - `training/train.py`: chuyển từ notebook hiện có thành script chạy được (ultralytics YOLO cls, tham số qua CLI: data dir, epochs, imgsz, batch, device). Hỗ trợ `--resume`.
  - `training/export_onnx.py`: nhận `--weights path.pt` (thay hardcode), export ONNX dynamic, in đường dẫn output.
  - `training/compute_sha256.py` (hoặc thêm vào export): in SHA256 của `.onnx` để dán vào `YOLODetector.MODEL_SHA256`.
  - `docs/`: hướng dẫn flow đầy đủ: kéo dataset HF → `prepare_dataset` → train cloud GPU → export → SHA256 → cập nhật `MODEL_SHA256` + `model download URL` → upload model lên HF → solver auto-download.
- Non-functional:
  - Script không yêu cầu GPU để *import*/test khô (test chỉ kiểm parse args + đường dẫn, không train thật).
  - Giữ verify SHA256 hiện có của solver (`detector/yolo_detector.py`) làm cổng an toàn cho model mới.

## Architecture

- `train.py`: hàm `train(data, epochs=50, imgsz=640, batch=64, device=0, ...)`, `if __name__=="__main__"` parse `argparse`/`click`. Output `runs/classify/<name>/weights/best.pt`.
- `export_onnx.py`: `argparse --weights`; `YOLO(weights).export(format="onnx", dynamic=True, half=False)`.
- `compute_sha256`: đọc file, `hashlib.sha256`, in hex — đối chiếu hằng `MODEL_SHA256` trong detector.
- Model versioning (CHỐT — Validation Session 1): tên file có version (vd `recaptcha_classification_57k_v2.onnx`) + sidecar `model_card.json` ghi `{date, epochs, imgsz, classes, sha256, dataset_size}`.
- Publish HF (CHỐT — Validation Session 1): **thủ công** (`huggingface-cli upload`), tài liệu hóa checklist; KHÔNG thêm `huggingface_hub` vào pipeline tự động vòng này.

## Related Code Files

- Create: `training/train.py` (từ `train_model.ipynb`)
- Modify: `training/export_onnx.py` (tham số hóa `--weights`)
- Create: `training/compute_sha256.py` (hoặc flag trong export)
- Read for context: `src/vision_ai_recaptcha_solver/detector/yolo_detector.py` (MODEL_SHA256, download URL, `_download_model`)
- Create: `docs/deployment-guide.md` mục "Retrain & publish model" (hoặc `docs/training-and-flywheel.md`)
- Create: `tests/test_training_scripts_args.py` (test khô: parse args, không train)

## Implementation Steps (TDD)

1. **Test trước (khô, không GPU):**
   - `tests/test_training_scripts_args.py`: import `train.py`/`export_onnx.py` không lỗi; parse args mặc định đúng; `export` từ chối weights không tồn tại; `compute_sha256` cho file tmp ra đúng hex độ dài 64.
2. Chuyển notebook → `train.py` (giữ hyperparams cũ: epochs=50, imgsz=640, batch=64, patience=15, amp=True, cache=True; `device` qua CLI).
3. Tham số hóa `export_onnx.py` (`--weights`).
4. Viết `compute_sha256`.
5. Viết tài liệu flow retrain→publish (kéo HF → prepare → train cloud → export → SHA256 → cập nhật detector → upload HF → auto-download).
6. ruff + mypy + pytest (test khô).
7. (Ngoài máy/manual) chạy thử 1 vòng train ngắn trên cloud GPU để xác nhận script — ghi lại kết quả; KHÔNG yêu cầu pass trong CI.

## Success Criteria

- [x] Test khô args/sha256 viết TRƯỚC và xanh (không cần GPU).
- [x] `train.py` chạy được cú pháp (smoke: `--help`); export nhận `--weights`.
- [x] `compute_sha256` khớp giá trị dán vào `MODEL_SHA256`.
- [x] Tài liệu flow retrain→publish đầy đủ, người khác lặp lại được.
- [x] Verify SHA256 của solver vẫn chặn model không khớp.
- [x] ruff + mypy + pytest xanh.

## Risk Assessment

- **Rủi ro:** không có GPU để verify train thật trong CI/local. **Giảm thiểu:** test khô + smoke; train thật làm thủ công cloud, ghi kết quả.
- **Rủi ro:** model mới đổi số/thứ tự class → solver mapping sai. **Giảm thiểu:** giữ thứ tự class cố định qua `class_mapping.py`; tài liệu cảnh báo; verify SHA256.
- **Rủi ro:** quên cập nhật download URL/SHA256 → solver kéo model cũ. **Giảm thiểu:** checklist publish trong docs + bước đối chiếu SHA256.
