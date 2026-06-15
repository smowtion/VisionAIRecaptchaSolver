---
phase: 3
title: "Repo Integration (dataset/training/class-mapping/review)"
status: completed
priority: P2
effort: "2-3d"
dependencies: [2]
---

# Phase 3: Repo Integration (dataset/training/class-mapping/review)

## Overview

Gộp `recaptcha-classification-57k` vào monorepo dưới `training/` (không ship trong wheel). Chuẩn hóa class-name (folder ↔ class_id ↔ solver label), viết `prepare_dataset.py` (gộp tile đã review vào dataset), và `review_cli.py` (hàng đợi gán nhãn thủ công cho `collected/`).

## Requirements

- Functional:
  - `training/` chứa: `class_mapping.py`, `prepare_dataset.py`, `review_cli.py`, `train.py` (Phase 4 hoàn thiện), `export_onnx.py` (di chuyển từ `train_model/`).
  - `class_mapping.py`: map 2 chiều folder-name ↔ solver class label, đồng bộ với `types.CLASS_NAMES`. Chuẩn hóa: `Stair→stairs`, `Hydrant→fire hydrant`, `Palm→palm tree`, `Traffic Light→traffic light`, ... `Other→other`.
  - `review_cli.py`: duyệt `collected/metadata.jsonl`, hiển thị từng tile (mở ảnh / in path), nhận nhãn từ người (chọn class hợp lệ hoặc `skip`/`discard`), ghi quyết định ra `collected/reviewed.jsonl`.
  - `prepare_dataset.py`: đọc `reviewed.jsonl`, copy tile đã gán nhãn vào `training/dataset/<train|val>/<Class>/` đúng tên chuẩn hóa, split train/val theo tỉ lệ cấu hình.
  - `train_model/` cũ được hợp nhất (xoá sau khi xác nhận `export_onnx.py` chạy ở vị trí mới).
- Non-functional:
  - `training/` KHÔNG vào wheel — verify `python -m build` không đóng gói `training/`.
  - Dataset raw (1.4GB) + `collected/` KHÔNG commit — `.gitignore` + ghi chú HF/git-lfs.
  - Mapping có test đảm bảo không lệch với `types.CLASS_NAMES`.

## Architecture

- `training/class_mapping.py`: `FOLDER_TO_LABEL: dict[str,str]`, `LABEL_TO_FOLDER`, hàm `normalize_folder(name)`; import/đối chiếu `vision_ai_recaptcha_solver.types.CLASS_NAMES` để fail nếu thiếu/đối lập.
- `review_cli.py` (KISS, dùng `click` đã có dependency): lệnh `review --collected-dir ... --out reviewed.jsonl`; lặp metadata chưa review, prompt nhãn, append quyết định.
- `prepare_dataset.py`: `merge --reviewed reviewed.jsonl --dataset training/dataset --val-split 0.1`.
- Cấu trúc thư mục cuối: như trong báo cáo brainstorm §"Monorepo layout".

## Related Code Files

- Create: `training/class_mapping.py`
- Create: `training/prepare_dataset.py`
- Create: `training/review_cli.py`
- Move: `train_model/export_onnx.py` → `training/export_onnx.py`; `train_model/train_model.ipynb` → `training/` (Phase 4 chuyển thành `train.py`)
- Modify: `.gitignore` (dataset raw, `collected/`, runs/)
- Modify: `pyproject.toml` (xác nhận `packages.find` chỉ gồm `vision_ai_recaptcha_solver*`; `training` không phải package)
- Create: `tests/test_class_mapping.py`, `tests/test_prepare_dataset.py`
- Modify: `docs/system-architecture.md`, `docs/codebase-summary.md` (mục training/flywheel)

## Implementation Steps (TDD)

1. **Test trước:**
   - `tests/test_class_mapping.py`: mọi folder dataset (`Car`,`Bridge`,`Stair`,`Hydrant`,`Palm`,`Traffic Light`,`Other`,...) map ra label hợp lệ; mọi target label trong `types.CLASS_NAMES` có folder tương ứng; round-trip `normalize_folder` ổn định.
   - `tests/test_prepare_dataset.py`: cho `reviewed.jsonl` giả + vài ảnh tmp → copy đúng vào `<Class>/`, split train/val đúng tỉ lệ, bỏ qua `discard`.
2. Viết `class_mapping.py` (đối chiếu `types.CLASS_NAMES`, fail nếu lệch).
3. Viết `prepare_dataset.py`.
4. Viết `review_cli.py` (click).
5. Di chuyển `export_onnx.py` sang `training/`; xác nhận import chạy; xoá `train_model/` sau khi ok.
6. Cập nhật `.gitignore` + verify `python -m build` không gói `training/`.
7. ruff + mypy + pytest.

## Success Criteria

- [x] Test mapping + prepare_dataset viết TRƯỚC và xanh.
- [x] `review_cli.py` gán nhãn được 1 mẫu thật từ `collected/` → `reviewed.jsonl`.
- [x] `prepare_dataset.py` gộp được vào `training/dataset/<Class>/`.
- [x] `python -m build` KHÔNG đóng gói `training/`.
- [x] Dataset raw + `collected/` không bị git theo dõi.
- [x] ruff + mypy + pytest xanh.

## Risk Assessment

- **Rủi ro:** class-name lệch âm thầm giữa folder và solver → dataset hỏng. **Giảm thiểu:** test đối chiếu bắt buộc với `types.CLASS_NAMES`.
- **Rủi ro:** vô tình đóng gói dataset/training vào wheel → PyPI phình. **Giảm thiểu:** verify build artifact trong success criteria.
- **Rủi ro:** commit nhầm dataset 1.4GB. **Giảm thiểu:** `.gitignore` trước, kiểm `git status` sạch.
