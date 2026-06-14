# Training & Data Flywheel

How the active-learning loop closes: the solver collects hard tiles → a human labels them →
they merge into the dataset → a cloud GPU retrains the classifier → the new ONNX is published
to Hugging Face → the solver auto-downloads it (gated by SHA256).

```
solve(collect_data=True)
   └─ DataCollector → collected/<date>/<type>/*.png + collected/metadata.jsonl
        └─ review_cli.py   (human labels each tile)        → collected/reviewed.jsonl
             └─ prepare_dataset.py (normalize + split)     → training/dataset/<train|val>/<Class>/
                  └─ train.py (cloud GPU)                  → runs/classify/<name>/weights/best.pt
                       └─ export_onnx.py                   → best.onnx
                            └─ compute_sha256.py           → <digest>
                                 └─ update YOLODetector.MODEL_SHA256 + MODEL_DOWNLOAD_URL
                                      └─ upload .onnx to Hugging Face
                                           └─ solver auto-downloads new model (SHA256 verified)
```

`training/` is **not** shipped in the PyPI wheel (`tool.setuptools.packages.find` only
includes `vision_ai_recaptcha_solver*`, and `training/` lives outside `src/`).

## 1. Enable collection (runtime)

Opt-in, disabled by default (zero I/O when off):

```python
from vision_ai_recaptcha_solver import RecaptchaSolver, SolverConfig

config = SolverConfig(collect_data=True, collect_dir="collected")
with RecaptchaSolver(config) as solver:
    solver.solve(website_key="...", website_url="...")
```

What gets collected (single write point: `collection/collector.py`):

| reason | when | source |
|--------|------|--------|
| `uncertain` | `min_confidence_threshold ≤ conf < conf_threshold` | per-tile, forwarded from `YOLODetector.classify_tiles_with_confidence` (tile already cropped — DRY) |
| `failed` | solve loop exhausts attempts → `TokenExtractionError` | solver |
| `unknown_keyword` | challenge keyword not in the class mapping | solver |

Output layout:

```
collected/
├── metadata.jsonl                     # one JSON object per sample/failure
└── <YYYY-MM-DD>/<captcha_type>/<pred_class>_<conf>_<uuid8>.png
```

`metadata.jsonl` fields: `ts, captcha_type, keyword, predicted_class, confidence, reason,
image_path, solve_outcome`. `collected/` is gitignored — never commit it.

## 2. Review (human labeling)

reCAPTCHA only returns pass/fail, so there is **no** per-tile ground truth — labels must come
from a human. No auto pseudo-labeling.

```bash
python training/review_cli.py --collected-dir collected --open
```

Walks unlabeled samples in `metadata.jsonl`, opens each tile (with `--open`), prompts for a
class (number / name) or `s`kip / `d`iscard / `q`uit, and appends decisions to
`collected/reviewed.jsonl`. Re-running resumes (already-reviewed images are skipped).

## 3. Merge into dataset

```bash
python training/prepare_dataset.py --reviewed collected/reviewed.jsonl \
    --dataset training/dataset --val-split 0.1
```

Copies kept tiles into `training/dataset/<train|val>/<CanonicalFolder>/`. Labels are
normalized via `training/class_mapping.py` (the single source of truth for
folder ↔ class_id ↔ solver label). `discard`/`skip` rows are ignored. The split is
stratified per class and deterministic for a given `--seed`.

> **Class order is fixed.** Folders map to class ids alphabetically (Bicycle=0 … Traffic
> Light=13), matching the model's embedded `names`. Validate any time with:
> `python training/class_mapping.py`.

## 4. Train (GPU: CUDA, Apple-Silicon MPS, or CPU)

`train.py` auto-detects the device (CUDA > MPS > CPU via `device_utils.resolve_device`), so
it runs on a cloud CUDA GPU **or** an Apple Silicon Mac (Metal/MPS) **or** CPU. Macs have no
CUDA but M-series chips train on MPS. Hyperparameters mirror the original notebook
(base `yolo11x-cls.pt`, epochs=50, imgsz=640, batch=64, patience=15, amp + cache on).

- **Small datasets** (e.g. the Tier B 4x4 set you collect+annotate): an Apple Silicon Mac
  (`--device mps`, or just `auto`) is fine — data stays local, no upload.
- **Large datasets** (e.g. a full 57k classification retrain): prefer a cloud CUDA GPU
  (Colab / cloud VM) — faster and frees the Mac.

```bash
python training/train.py --data training/dataset                  # auto device
python training/train.py --data training/dataset --device mps --no-amp   # Apple Silicon
python training/train.py --data training/dataset --device 0       # explicit CUDA
python training/train.py --resume                                 # continue from best.pt
```

> On MPS, pass `--no-amp` if mixed precision misbehaves (some ops lack MPS kernels).

Output: `runs/classify/rec_cls_model/weights/best.pt`.

## 5. Export ONNX + SHA256

```bash
python training/export_onnx.py --weights runs/classify/rec_cls_model/weights/best.pt
python training/compute_sha256.py runs/classify/rec_cls_model/weights/best.onnx
```

`export_onnx.py` writes `best.onnx` (dynamic axes, fp32). `compute_sha256.py` prints the
64-char digest to paste into the detector.

## 6. Version, update detector, publish

**Versioning (decided):** version the filename and ship a sidecar `model_card.json`.

```
recaptcha_classification_57k_v2.onnx
recaptcha_classification_57k_v2.model_card.json   # {date, epochs, imgsz, classes, sha256, dataset_size}
```

**Update the solver** (`src/vision_ai_recaptcha_solver/detector/yolo_detector.py`):

1. Set `MODEL_SHA256` to the new digest from step 5.
2. Point `MODEL_DOWNLOAD_URL` at the new Hugging Face file.

The solver verifies every download against `MODEL_SHA256` and rejects a mismatch — this is
the safety gate. If you forget to update the digest, downloads fail loudly rather than
shipping a wrong/corrupt model.

**Publish to Hugging Face (manual — decided; not automated this round):**

```bash
huggingface-cli login
huggingface-cli upload DannyLuna/recaptcha-classification-57k \
    recaptcha_classification_57k_v2.onnx
huggingface-cli upload DannyLuna/recaptcha-classification-57k \
    recaptcha_classification_57k_v2.model_card.json
```

### Publish checklist

- [ ] `python training/class_mapping.py` passes (class order unchanged).
- [ ] New model's class count + order match `class_mapping.FOLDER_ORDER` (14 classes).
- [ ] `MODEL_SHA256` updated to the new digest.
- [ ] `MODEL_DOWNLOAD_URL` updated to the new HF file.
- [ ] `.onnx` + `model_card.json` uploaded to Hugging Face.
- [ ] Fresh `pip install` (no cached model) downloads the new model and a solve smoke-test
      passes.

> **Warning:** if a retrain changes the number or order of classes, the runtime mapping in
> `types.py` / `class_mapping.py` must change in lockstep, or the solver will click the wrong
> tiles. Keep `FOLDER_ORDER` and the model's `names` aligned.

## Tier B: custom 4x4 detection model (COCO gap)

The bundled COCO model (`yolo12x.pt`) lacks 7 reCAPTCHA classes — bridges, chimneys,
crosswalks, mountains or hills, palm trees, stairs, tractors. For those, 4x4 currently uses
the **per-cell classification fallback** (`SquareCaptchaHandler._classify_cells_fallback`).
Tier B trains a dedicated **detection** model on these classes (better for one large object
spanning cells) via a separate bbox pipeline. The 4x4 handler runs a 3-tier priority:
COCO detection → custom detection (if a model is loaded) → per-cell fallback.

> Detection needs **full-image + bounding-box** data, which the per-cell classification
> flywheel does NOT produce. Tier B has its own collection + annotation pipeline.

1. **Collect full images** — enable collection; `DataCollector.record_challenge_image`
   saves whole 4x4 images to `collected/full/` + `collected/full/metadata.jsonl`.
2. **Annotate (cell → bbox)** — `python training/annotate_detection_cli.py --collected-dir
   collected/full --open`: pick class + cells (1..16); each selected cell becomes one YOLO
   box (cell-level weak supervision) → `annotations.jsonl`.
3. **Build detection dataset** — `python training/prepare_detection_dataset.py --annotations
   collected/full/annotations.jsonl --dataset training/detection_dataset`: emits
   `images/`, `labels/`, `data.yaml` (names = `class_mapping.DETECTION_CLASSES`).
4. **Train (CUDA / Apple-Silicon MPS / CPU)** — `python training/train_detection.py --data
   training/detection_dataset/data.yaml` (auto device; base `yolo11x.pt`, task detect). The
   Tier B set is small, so an Apple Silicon Mac (`--device mps --no-amp`) is a fine choice —
   no dataset upload. Use a cloud CUDA GPU if it grows large. (Phase 3: `train_detection.py`
   reuses `device_utils.resolve_device` + the `--amp/--no-amp` flag, like `train.py`.)
5. **Export + SHA256 + publish** — `export_onnx.py --weights .../best.pt`,
   `compute_sha256.py best.onnx`, then set `YOLODetector.CUSTOM_DETECTION_MODEL_URL` +
   `CUSTOM_DETECTION_SHA256`, upload to Hugging Face.
6. **Enable at runtime** — `SolverConfig(custom_detection_model_path="...onnx")`. With no
   path set (default) the runtime is unchanged (COCO + per-cell fallback).

> **Contract:** `types.CUSTOM_DETECTION_CLASSES` (runtime) MUST equal
> `training/class_mapping.DETECTION_CLASSES` (training) — a test enforces this. A mismatch
> maps detections to the wrong class.
