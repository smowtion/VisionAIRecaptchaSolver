# Vision AI reCAPTCHA Solver — Codebase Summary

## Overview

Vision AI reCAPTCHA Solver is a ~4,400-line Python library (src/) organized into modular packages with clear separation: browser automation, YOLO detection, captcha handler dispatch, configuration, and resource management. Total project ~4,600 lines including tests and demos.

---

## Core Modules

### Main Package: `vision_ai_recaptcha_solver/`

#### Public API & Exports (`__init__.py` — 66 LOC)
**Responsibility:** Re-export public symbols; establish package contract.
**Exports:** `RecaptchaSolver`, `AsyncRecaptchaSolver`, `SolverConfig`, `SolveResult`, `DetectionResult`, `CaptchaType`, `CLASS_NAMES`, `TARGET_MAPPINGS`, `COCO_TARGET_MAPPINGS`, exception classes, `__version__`.
**Note:** All public symbols defined elsewhere; this file is purely a barrel export for IDE/user convenience.

#### Configuration (`config.py` — 194 LOC)
**Responsibility:** Configuration dataclass with validation.
**Key Class:** `SolverConfig`
- Attributes: `model_path`, `detection_model_path`, `download_dir`, `server_port`, `proxy`, `browser_path`, `headless`, `timeout`, `max_attempts`, `human_delay_mean/sigma`, `log_level`, `low_confidence_threshold`.
- Validation: `__post_init__` runs regex on proxy URL, checks port range (1–65535), validates timeout > 0, ensures browser path exists.
- Sentinel pattern: `_UNSET` object distinguishes explicit vs. defaulted `server_port` and `download_dir` (drives resource allocation).
- No defaults for `download_dir` / `server_port`; auto-allocated by solver if not set.

#### Type Definitions (`types.py` — 392 LOC)
**Responsibility:** Enum and dataclass definitions; multilingual mappings.
**Key Types:**
- `CaptchaType` enum: `DYNAMIC_3X3`, `SELECTION_3X3`, `SQUARE_4X4`, `INVISIBLE`, `NO_CHALLENGE`, `UNKNOWN`.
- `SolveResult`: frozen dataclass with `token`, `cookies`, `time_taken`, `captcha_type`, `attempts`.
- `DetectionResult`: frozen dataclass with `answers` (grid indices), `confidence`, `target_class`.
- `CLASS_NAMES`: list of dicts mapping English class names to multilingual synonyms (8 languages).
- `TARGET_MAPPINGS`: dict mapping class names to YOLO class IDs (for classification model).
- `COCO_TARGET_MAPPINGS`: dict mapping class names to COCO class IDs (for detection model).
- Helper functions: `get_target_keyword()` (extract challenge noun), `_build_multilang_mappings()`.

#### Synchronous Solver (`solver.py` — 586 LOC)
**Responsibility:** Main solve entrypoint; lifecycle management; signal handling.
**Key Class:** `RecaptchaSolver`
- Context manager: `__enter__`, `__exit__` for safe cleanup.
- `solve(website_key, website_url, is_invisible=False, action="verify", is_enterprise=False)`: End-to-end solve pipeline.
  1. Instantiate `RecaptchaDomainReplicator`, start local HTTPS server.
  2. Launch Chromium via replicator.
  3. Click checkbox (reCAPTCHA v2) or wait for invisible challenge (v3).
  4. Dispatch to handler based on challenge type (3x3 dynamic/selection or 4x4 square).
  5. Extract token from replicator.
  6. Return `SolveResult`.
- Signal handling: Registers `SIGINT`/`SIGTERM` handlers on first instance (guarded by `_cleanup_registered`); tracked in `WeakSet(_live_solvers)`.
- Resource management: Calls `reserve_solver_resources()` on init; `release_solver_resources()` on close.
- Cleanup: Idempotent `close()` method; temp dir cleanup gated on `_owns_download_dir` (marker file `.vision_ai_recaptcha_solver_owned`).

#### Asynchronous Solver (`async_solver.py` — 569 LOC)
**Responsibility:** Async variant of `RecaptchaSolver`.
**Key Class:** `AsyncRecaptchaSolver`
- Parallel implementation (not a wrapper) of `solver.py`.
- Browser calls offloaded to `ThreadPoolExecutor` to avoid blocking event loop.
- Identical API shape: `async with AsyncRecaptchaSolver(config) as solver: result = await solver.solve(...)`.
- When changing core solve logic, update both `solver.py` and `async_solver.py`.

#### Exceptions (`exceptions.py` — 80 LOC)
**Responsibility:** Exception hierarchy.
**Base Class:** `RecaptchaSolverError` (all custom exceptions inherit).
**Exception Classes:**
- `BrowserError`, `CaptchaNotFoundError`, `UnsupportedCaptchaError`, `DetectionError`, `TokenExtractionError`, `SolverTimeoutError`, `ModelNotFoundError`, `ImageDownloadError`, `LowConfidenceError`, `ElementNotFoundError`, `NavigationError`.
- `CaptchaTimeoutError` is an alias of `SolverTimeoutError` (kept for backward compatibility; do not remove).

#### Logging Configuration (`logging_config.py` — 45 LOC)
**Responsibility:** Centralized logging setup.
**Function:** `setup_logging(log_level: str)` — configures root logger, filters external library noise.

#### Resource Allocation (`resource_allocation.py` — 103 LOC)
**Responsibility:** Serialize per-solver resource claims.
**Mechanism:** Module-global `WeakKeyDictionary` under `threading.Lock`.
**Functions:**
- `reserve_solver_resources(solver, server_port, download_dir)`: Auto-allocates free port/unique dir if not set; warns if explicit value conflicts with in-use resource.
- `release_solver_resources(solver)`: Removes solver from allocations.
**Concurrency:** Safe for 5+ concurrent solver instances per machine.

#### Constants & Utilities
- `constants.py` (22 LOC): `DEFAULT_SERVER_PORT`, `DEFAULT_DOWNLOAD_DIR`, `VALID_LOG_LEVELS`.
- `utils.py` (26 LOC): `human_delay()` function for realistic timing.
- `__main__.py` (198 LOC): Click CLI with `demo` and `solve` commands.

---

## Browser Automation Module: `browser/`

### Navigation (`browser/navigation.py` — 515 LOC)
**Responsibility:** Chromium browser interaction.
**Key Functions:**
- `click_checkbox(page, timeout)`: Click the reCAPTCHA checkbox.
- `get_challenge_iframe(page, timeout)`: Locate challenge iframe in DOM.
- `get_challenge_title(page, timeout)`: Extract challenge title (e.g., "Select all fire hydrants").
- `get_target_keyword(title)`: Parse English keyword from title (multilingual via `CLASS_NAMES`).
- `click_tile(page, tile_index, timeout)`: Click a specific tile in the grid.
- `click_verify_button(page, timeout)`: Submit the challenge.
- `is_solved(page)`: Check if token has been extracted.
- `wait_for_verify_result(page, timeout)`: Poll for verify completion.
- `click_reload_button(page, timeout)`: Reload challenge (for dynamic type).

**Chromium Integration:** Uses `PlaywrightBrowser` (via `recaptcha-domain-replicator`) for reliable DOM interaction.

---

## Detection Module: `detector/`

### YOLO Detector (`detector/yolo_detector.py` — 651 LOC)
**Responsibility:** Image inference via two YOLO models.
**Key Class:** `YOLODetector`
- **Classification Model (ONNX):** `recaptcha_classification_57k.onnx` (14 classes).
  - Auto-downloaded from Hugging Face on first use.
  - SHA256 verified: `4092e8917ee8c2963895d66ba10a97d6ef975c468a95858a8a7bd9e70681b65d`.
  - Used for 3x3 challenges.
- **Detection Model (PyTorch):** `yolo12x.pt` (COCO 80 classes).
  - Auto-downloaded by `ultralytics` library.
  - Used for 4x4 challenges.
- **Warmup:** Automatic model warmup in background thread on first instantiation.
- **Methods:**
  - `detect(image, captcha_type)`: Run inference; return `DetectionResult` with grid indices, confidence, class ID.
  - `warm_up()`: Load models into memory (called in background).

**Performance:** ~1s per 3x3 image; ~2s per 4x4 image (hardware-dependent).

### Grid Utils (`detector/grid_utils.py` — 113 LOC)
**Responsibility:** Geometry calculations for grid splitting.
**Key Functions:**
- `calculate_3x3_cells(image_width, image_height)`: Return 9 cell bounding boxes.
- `calculate_4x4_cells(image_width, image_height)`: Return 16 cell bounding boxes.

---

## Captcha Handler Module: `captcha/`

### Base Handler (`captcha/base_handler.py` — 116 LOC)
**Responsibility:** Abstract base for challenge-specific logic.
**Key Class:** `BaseCaptchaHandler` (ABC)
- **Abstract method:** `handle(page, detector, attempts_remaining)`: Dispatch to browser/detector interaction.
- **Lifecycle:** Handlers are instantiated once per solve; torn down after challenge completion.

### Dynamic Handler (`captcha/dynamic_handler.py` — 313 LOC)
**Responsibility:** Handle 3x3 dynamic challenges (tiles refresh after each click).
**Key Class:** `DynamicCaptchaHandler(BaseCaptchaHandler)`
- Prediction → click tile → wait for reload → repeat until solved.
- Tracks `attempts` to avoid infinite loops.

### Selection Handler (`captcha/selection_handler.py` — 86 LOC)
**Responsibility:** Handle 3x3 static selection challenges.
**Key Class:** `SelectionCaptchaHandler(BaseCaptchaHandler)`
- Prediction → click all predicted tiles → submit once.

### Square Handler (`captcha/square_handler.py`)
**Responsibility:** Handle 4x4 square challenges.
**Key Class:** `SquareCaptchaHandler(BaseCaptchaHandler)`
- Primary: COCO detection on the full image (`detect_for_grid`, `GRID_SIZE=450` px).
- Fallback (`_classify_cells_fallback`): when the keyword is not a COCO class
  (stairs/bridges/crosswalks/chimneys/mountains/palm/tractor), split into 16 cells and
  classify each with the 57k model (`classify_tiles_with_confidence`, `GRID_CELLS=4`),
  keep cells with conf ≥ `conf_threshold`. Covers all 14 classes for 4x4.

### Solve robustness (fail-fast + speed)
- `YOLODetector.is_supported(keyword, captcha_type)`: 4x4 = COCO **or** classification;
  3x3 = classification. Drives a fast-skip in the solve loop (both solvers): unsupported
  challenges are cheap-reloaded under a separate `skips` budget (`max_attempts*3`) instead
  of burning real attempts. `_reload_challenge(fast=)` trims delay on the skip path only.
- On solve failure the token wait drops from the full `timeout` to `default_timeout` so a
  failed solve returns fast instead of hanging. Implemented symmetrically in
  `solver.py` + `async_solver.py`.

### Image Utils (`captcha/image_utils.py` — 160 LOC)
**Responsibility:** Image processing helpers.
**Key Functions:**
- `split_image_into_cells(image, rows, cols)`: Numpy array splitting.
- `get_cell_image(image, cell_index, rows, cols)`: Extract single cell.

---

## Data Collection Module: `collection/` (opt-in active learning)

### Data Collector (`collection/collector.py`)
**Responsibility:** Single write point for the active-learning data flywheel. Disabled by
default (`SolverConfig.collect_data=False`) → zero I/O. When enabled, persists hard samples
to `collect_dir` for human review.
**Key API:**
- `record_tile(image, cell, confidence, *, predicted_class, captcha_type, keyword)`: saves a
  tile only when in the uncertain band (`min_confidence_threshold ≤ conf < conf_threshold`).
- `record_failure(captcha_type, keyword, reason, images=None)`: records `failed` /
  `unknown_keyword` outcomes.
- `set_context(captcha_type, keyword)`: per-solve context used by the detector tile hook.
**Wiring:** injected into `YOLODetector(collector=...)` (tile hook reuses already-cropped
tiles in `classify_tiles_with_confidence` — DRY); failure hooks live in both `solver.py` and
`async_solver.py` (`_get_target_class(browser, captcha_type)`). Async disk writes offloaded
via `_run_in_executor`. Output: `collected/<date>/<type>/<cls>_<conf>_<uuid8>.png` +
`collected/metadata.jsonl`. Best-effort: never raises into the solve loop.

## Training & Flywheel: `training/` (not shipped in wheel)

Outside `src/` (excluded from the PyPI wheel). Closes the loop: collect → review → merge →
train → export → publish → auto-download. See `docs/training-and-flywheel.md` for the full flow.
- `class_mapping.py`: source of truth for folder ↔ class_id ↔ solver label (14 classes,
  validated against `types.CLASS_NAMES`).
- `review_cli.py`: human labeling queue (`metadata.jsonl` → `reviewed.jsonl`).
- `prepare_dataset.py`: merge reviewed tiles into `training/dataset/<train|val>/<Class>/`.
- `train.py` / `export_onnx.py` / `compute_sha256.py`: cloud-GPU train → ONNX → SHA256 gate.

---

## Testing

### Unit Tests (`tests/`)
- `test_config.py`: Config validation, sentinel pattern.
- `test_types.py`: Type conversions, CLASS_NAMES mapping.
- `test_resource_allocation.py`: Resource claim serialization.
- `test_grid_utils.py`: Cell geometry calculations.
- `test_image_utils.py`: Image splitting logic.

### Integration Tests (`tests/integration/`)
- `test_google_demo.py` (opt-in via `pytest -m integration`): Solve Google's public reCAPTCHA demo headless; verify token extraction.
- Excluded from default run (`pytest`) via `addopts = "-m 'not integration'"` in `pyproject.toml`.

### Demo Scripts
- `demo.py`: Synchronous example against Google demo.
- `demo_async.py`: Asynchronous example.

---

## File Size Analysis

| File | LOC | Category | Notes |
|------|-----|----------|-------|
| detector/yolo_detector.py | 651 | Core | Candidate for modularization (model DL, hash verify, warmup) |
| solver.py | 586 | Core | Candidate for modularization (nav, handlers dispatch, cleanup) |
| async_solver.py | 569 | Core | Parallel to solver.py; keep in sync |
| browser/navigation.py | 515 | Core | Candidate for split (find, click, wait utilities) |
| types.py | 392 | Core | Dense but manageable (enums + multilingual mappings) |
| captcha/dynamic_handler.py | 313 | Handler | Self-contained |
| __main__.py | 198 | CLI | Self-contained |
| config.py | 194 | Config | Self-contained |
| captcha/image_utils.py | 160 | Utils | Self-contained |
| captcha/base_handler.py | 116 | Handler | Self-contained |
| detector/grid_utils.py | 113 | Utils | Self-contained |
| resource_allocation.py | 103 | Core | Self-contained; critical for concurrency |
| captcha/selection_handler.py | 86 | Handler | Self-contained |
| captcha/square_handler.py | 84 | Handler | Self-contained |
| exceptions.py | 80 | Core | Self-contained |
| logging_config.py | 45 | Utils | Self-contained |
| browser/__init__.py | 31 | Package | Imports |
| utils.py | 26 | Utils | Self-contained |
| captcha/__init__.py | 23 | Package | Imports |
| constants.py | 22 | Constants | Self-contained |
| detector/__init__.py | 15 | Package | Imports |
| __init__.py | 66 | Package | Public API barrel |
| **Total** | **~4,388** | | Across src/ only |

**Observation:** Four files exceed 500 LOC. While modularization is a future goal (see roadmap), current structure prioritizes correctness and feature completeness over splitting. Fine-grained modules will be extracted as technical debt reduction effort.

---

## Dependency Graph

```
RecaptchaSolver
  ├─ RecaptchaDomainReplicator (external)
  ├─ browser.navigation (Chromium/Playwright)
  ├─ detector.YOLODetector
  │   ├─ ultralytics (YOLO detection)
  │   ├─ onnxruntime (classification inference)
  │   ├─ opencv-python (image resize/norm)
  │   └─ numpy
  ├─ captcha.{DynamicHandler, SelectionHandler, SquareCataHandler}
  │   ├─ detector.grid_utils
  │   ├─ captcha.image_utils
  │   └─ browser.navigation
  └─ resource_allocation (threading.Lock + WeakKeyDictionary)

SolverConfig
  └─ constants (validation bounds)
```

---

## Patterns & Conventions

### 1. Future Annotations
All modules start with `from __future__ import annotations` (enforced by style).

### 2. Public API Barrel
Only `__init__.py` re-exports public symbols; consumers must use `from vision_ai_recaptcha_solver import RecaptchaSolver` (not `from ...solver import RecaptchaSolver`).

### 3. Sentinel for Optional Config
`_UNSET` object used instead of `None` or `Optional` to allow explicit `None` values and distinguish user intent (see `SolverConfig`).

### 4. Parallel Sync/Async
`solver.py` and `async_solver.py` are independent, not wrappers. When updating core logic, both must be updated.

### 5. Exception Hierarchy
All custom exceptions inherit `RecaptchaSolverError`; code can `except RecaptchaSolverError` to catch all custom errors.

### 6. Resource Lifecycle
Solvers are tracked in `WeakSet`; cleanup is automatic when last reference drops. Manual `close()` is also available and idempotent.

### 7. Multilingual Mapping
`CLASS_NAMES` is the single source of truth; `TARGET_MAPPINGS` and `COCO_TARGET_MAPPINGS` derive from it automatically. Adding a language synonym propagates to all mapping dicts.

---

## Known Technical Debt

1. **Large core files** — `yolo_detector.py` (651), `solver.py` (586), `async_solver.py` (569), `navigation.py` (515) exceed modularization target. See roadmap for extraction candidates.
2. **Parallel implementations** — Sync/async are separate. Shared logic could be extracted into utility modules (low priority; current duplication is maintainable).
3. **Browser session reuse** — Currently one session per solver instance; future work could pool sessions for repeated solves.

---

## Last Updated

2026-06-13
