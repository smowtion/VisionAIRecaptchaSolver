---
phase: 2
title: Cell-bbox annotation + detection dataset
status: completed
priority: P1
effort: 1-1.5d
dependencies:
  - 1
---

# Phase 2: Cell-bbox annotation + detection dataset

## Overview

Annotate ảnh 4x4 full (Phase 1) thành nhãn YOLO detect, rồi build dataset detection (images/ + labels/ + data.yaml). Dùng **cell-level bbox** (weak supervision): người chọn các ô (1–16) chứa object → mỗi ô thành 1 bbox (toạ độ ô trên lưới 4x4). Nhanh để annotate, đủ cho vòng đầu.

## Requirements

- Functional:
  - `training/annotate_detection_cli.py` (click): duyệt `collected/full/metadata.jsonl`, mở ảnh, người nhập class + danh sách ô chứa object (vd "stairs 1,2,5"); ghi `collected/full/annotations.jsonl` `{image_path, label, cells:[...]}`. Resume được (bỏ ảnh đã annotate).
  - `training/prepare_detection_dataset.py`: đọc annotations → với mỗi ảnh ghi `labels/<name>.txt` (mỗi dòng `class_id cx cy w h` chuẩn YOLO, normalized) từ cell index (ô k trên lưới 4 → bbox ô đó); copy ảnh vào `images/`; split train/val; sinh `data.yaml` (names theo class_mapping detection).
  - Class id detection: thứ tự cố định cho 7 lớp thiếu (hoặc full reCAPTCHA classes) — định nghĩa trong `training/class_mapping.py` (mở rộng: `DETECTION_CLASSES` ordered list) để folder↔id↔label đồng bộ.
- Non-functional:
  - Cell→bbox toán đúng (ô k: row=k//4, col=k%4 → cx=(col+0.5)/4, cy=(row+0.5)/4, w=h=1/4). Test bắt buộc.
  - dataset/ảnh raw KHÔNG commit (.gitignore).

## Architecture

- `training/class_mapping.py`: thêm `DETECTION_CLASSES: list[str]` (ordered) + `DETECTION_LABEL_TO_ID`. Bao 7 lớp COCO-thiếu (mở rộng full sau).
- `annotate_detection_cli.py`: KISS, click, mở ảnh bằng OS viewer (tái dùng `_open_externally` pattern của review_cli).
- `prepare_detection_dataset.py`: `cell_to_yolo_bbox(cell, grid=4) -> (cx,cy,w,h)`; build YOLO detect layout + data.yaml.

## Related Code Files

- Create: `training/annotate_detection_cli.py`
- Create: `training/prepare_detection_dataset.py`
- Modify: `training/class_mapping.py` (DETECTION_CLASSES + id map)
- Modify: `.gitignore` (training/detection_dataset/)
- Create: `tests/test_detection_dataset.py` (cell→bbox, prepare, data.yaml)
- Modify: `tests/test_class_mapping.py` (DETECTION_CLASSES consistency)

## Implementation Steps (TDD)

1. **Test trước:** `test_detection_dataset.py` — `cell_to_yolo_bbox` đúng cho ô góc/giữa (vd ô 1→(0.125,0.125,0.25,0.25)); `prepare_detection_dataset` từ annotations giả → labels/*.txt đúng dòng + images copy + data.yaml có names đúng + split. `test_class_mapping`: DETECTION_CLASSES không trùng, id liên tục 0..n-1.
2. Mở rộng `class_mapping.py` (DETECTION_CLASSES).
3. Viết `prepare_detection_dataset.py` (cell→bbox + builder + data.yaml).
4. Viết `annotate_detection_cli.py`.
5. `.gitignore` + ruff + mypy + pytest.

## Success Criteria

- [ ] Test cell→bbox + prepare + class_mapping viết TRƯỚC và xanh.
- [ ] `prepare_detection_dataset` sinh YOLO detect dataset hợp lệ (images/labels/data.yaml).
- [ ] `annotate_detection_cli` annotate được 1 ảnh thật → annotations.jsonl.
- [ ] dataset không bị git theo dõi; ruff + mypy clean.

## Risk Assessment

- **Cell-bbox thô (weak supervision):** chấp nhận vòng đầu; doc cảnh báo; có thể nâng bbox chặt sau (annotation tool ngoài).
- **Lệch class id detection vs solver mapping:** test class_mapping đối chiếu bắt buộc.
- **Toạ độ normalize sai → nhãn rác:** unit test cell→bbox cố định.
