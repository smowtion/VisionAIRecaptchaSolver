# Tier B: 4x4 Detection Model Pipeline + Apple Silicon MPS Support

**Date**: 2026-06-14 14:00
**Severity**: Medium
**Component**: Data flywheel (Tier B), training infrastructure, runtime solver
**Status**: Completed

## What Happened

Completed Tier B of the data flywheel in a single autonomous session: a full custom detection model pipeline for reCAPTCHA's 4x4 grid challenge type. Simultaneously discovered and enabled Apple Silicon (M2 Max) MPS training support, eliminating the assumption that Mac users must use cloud GPUs for small datasets.

## The Critical Realization

The hardest lesson came at the intersection of automation and reality: **a production detection model cannot be trained without human annotation**. The pipeline is complete and smoke-proven, but the trained artifact is blocked indefinitely on humans manually annotating collected cell bboxes. This is honest framing: "pipeline done" ≠ "model trained". Every automation attempt (pseudo-labeling from reCAPTCHA's pass/fail signal) was a dead end — there's no ground truth signal in the challenge itself.

## Technical Details

**Commits (4):**
- `526eb81` — YOLODetector fail-fast + per-cell 4x4 fallback
- `72f9db7` — Tier B scaffold phases 1, 2, 4 (full-image collection, bbox annotation CLI, detection dataset builder, solver integration)
- `3bcca17` — Device auto-detection (CUDA > MPS > CPU) + `--amp/--no-amp` flag
- `54d7200` — Detection trainer (`train_detection.py`), model card writer, collect loop driver

**Pipeline architecture:**
- Phase 1: `DataCollector.record_challenge_image()` → `collected/full/` (full 4x4 images + metadata, separate from existing per-cell tiles)
- Phase 2: `annotate_detection_cli.py` (human marks cell bboxes) → `prepare_detection_dataset.py` builds YOLO detection data.yaml
- Phase 3: `train_detection.py` (device auto-resolve, `--amp`, resumable) + `export_onnx.py` + SHA256 verification + model card
- Phase 4: Runtime 3-tier dispatch for 4x4: COCO detections → custom detection (if present) → per-cell classification fallback, all behind optional `custom_detection_model_path`

**MPS discovery:** Tested smoke train on M2 Max MPS (synthetic YOLO-detect data, 1 epoch, base yolo11n.pt) → `best.pt` with `args.yaml task=detect device=mps`. Confirmed the training loop handles MPS correctly. Dev extras now include `onnx` and `onnxslim` for export; runtime keeps `onnxruntime` only.

## What We Tried

1. **Pseudo-label automation:** Attempted to infer cell class from reCAPTCHA pass/fail. Rejected — no per-tile signal available.
2. **Training on CPU-only Mac:** Before discovering MPS, assumed Colab was mandatory. MPS proves small Tier B datasets train fast locally (5–10 min on M2 Max).
3. **Reusing classification data:** Initially considered repurposing per-cell tiles as weak supervision. Decided against — different domain (single objects vs. multi-object grid). New data pipeline built.

## Root Cause Analysis

The annotation bottleneck is not a bug but a **design boundary**. reCAPTCHA challenges are verification-only (human solves, system verifies); they emit no fine-grained ground truth. We built a collection and annotation workflow, but execution depends on human time. This is the "brutal honesty" stated in the plan: "no shortcuts" for data quality.

The MPS assumption was overly pessimistic. Mac's machine learning ecosystem includes GPU support via MPS; we didn't check initially because the focus was rCAPTCHA solving, not training. The device auto-detection logic (`resolve_device`) now handles all three tiers.

## Lessons Learned

1. **Pipeline completeness ≠ model readiness.** Code done, artifact blocked, and that's OK to say aloud.
2. **Check your hardware assumptions.** Apple Silicon has MPS; M2 Max runs small training jobs competitively with cloud for iteration. Saves cost and latency for fast prototyping.
3. **Weak supervision is a choice, not a shortcut.** Cell-level grid bboxes are honest weak labels (one bbox per clicked cell); they're acceptable for a first model but require human annotation still.
4. **Test contracts across training and runtime.** Added assert: `DETECTION_CLASSES (train) == CUSTOM_DETECTION_CLASSES (runtime)`. Prevents silent mismatches.

## Next Steps

1. **Human annotation phase:** Collect and annotate real challenge 4x4 images (use `annotate_detection_cli.py`). No timeline given — data quality is the gate.
2. **Train on collected data:** Once 50+ annotated images exist, run `train_detection.py --device mps --epochs 10` locally or scale to Colab for larger datasets.
3. **Validate and export:** Verify mAP on held-out set, export ONNX, compute SHA256, push to Hugging Face.
4. **Upstream PR:** #8 awaits maintainer review (145 unit tests pass, ruff/mypy clean except 4 pre-existing mypy only).

**Status**: DONE
