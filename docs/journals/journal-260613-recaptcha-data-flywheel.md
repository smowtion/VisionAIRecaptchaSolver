# Data Flywheel: 4-Phase Cook Execution Complete

**Date**: 2026-06-13 23:10  
**Severity**: Medium  
**Component**: solver + detector + collection module  
**Status**: Resolved  

## What Happened

Four-phase implementation of active-learning data collection pipeline for reCAPTCHA solver (commit `824f8ff`). Added opt-in `DataCollector` to capture uncertain/failed tiles for human review, feeding a training loop that re-exports ONNX models. All 107 tests pass; public API unchanged; wheel excludes training code.

## The Brutal Truth

This was a clean execution — no surprises, no fires. The plan was thorough (pre-verification caught design changes before code), and the team wrote tests before features. That meant code review found a subtle but critical bug that testing missed entirely: exception handling in a telemetry path that **must never abort the solve**.

## Technical Details

**Collector architecture:**
- `collection/DataCollector` writes PNG tiles + `metadata.jsonl` (reasons: `uncertain` ≤ confidence < threshold, `failed` no tile match, `unknown_keyword` unmapped class)
- Hook placed in `YOLODetector.classify_tiles_with_confidence` (line ~518) to reuse already-cropped tiles (DRY principle)
- Wired symmetrically into both `RecaptchaSolver` and `AsyncRecaptchaSolver` (parallel impls, not wrappers)
- Async disk writes offloaded via `_run_in_executor` to avoid blocking event loop
- Config flag `collect_data=False` by default → zero I/O overhead for PyPI users

**Training tooling (outside wheel):**
- `training/class_mapping.py` — single source of truth: folder ↔ class_id ↔ label (14 classes, validated vs `types.CLASS_NAMES`)
- `prepare_dataset.py`, `review_cli.py`, `train.py`, `export_onnx.py`, `compute_sha256.py`
- Excluded from wheel via `tool.setuptools.packages.find where=src` (training/ lives at root)

## What We Tried

Wrote tests first per TDD mode, blocking all new code:
- `test_config.py` (+7) — config sentinel & thresholds
- `test_collector_scaffold.py` — no-op disabled collector
- `test_data_collector.py` — tile I/O, metadata format
- `test_class_mapping.py` — class id/label round-trip
- `test_prepare_dataset.py` — dataset preparation
- `test_training_scripts_args.py` — script CLI args (dry run, no GPU)

CI green: 107 passed (+38 new), ruff clean on `src/` + `training/`, mypy `src/` showing only 4 pre-existing errors on HEAD.

## Root Cause Analysis (the Hard Lesson)

Code review flagged a narrow exception handler that nearly shipped:

```python
try:
    cv2.imwrite(tile_path, tile)  
except OSError:  # WRONG
    logger.warning("failed to write tile")
```

`cv2.error` (from `cv2.imwrite`) is **not** an `OSError` subclass. A corrupt OpenCV environment would raise `cv2.error`, bypass the handler, and propagate into `classify_tiles_with_confidence`, breaking the solve pipeline for the user. Telemetry must **never** abort the primary flow.

**Fixed to:**
```python
except Exception:  # catch ALL, never abort
    logger.warning("failed to write tile")
```

Tests passed because test environment had healthy OpenCV. The bug only surfaces in edge cases (missing codec, corrupted install, file system full on unknown error). Code review caught it; tests didn't.

## Lessons Learned

1. **Telemetry/observability code must be defensive.** If the feature is "nice to have" (data collection), wrap it in a broad exception handler. Narrow catches (`OSError`) assume the stdlib exception hierarchy is stable; it's not (NumPy, OpenCV, Pillow each have their own exception trees).

2. **TDD locks behavior, but doesn't catch all bugs.** Tests verify the happy path and specified error cases. They don't enumerate all possible exception types the third-party libs might throw. Code review with domain knowledge (knowing `cv2.error` exists) caught what tests missed.

3. **Symmetry matters.** Because `RecaptchaSolver` and `AsyncRecaptchaSolver` are **parallel implementations, not wrappers**, every logic change must land in both. Phase 1 wiring + Phase 2 hooks went into both without friction — the pattern worked.

4. **Hook placement at the detector level (DRY).** The detector already crops tiles; asking handlers to re-crop them is waste. Putting the collection hook in `classify_tiles_with_confidence` reused existing context, reduced code paths, and simplified testing.

## Next Steps

- Monitor production for telemetry failures (won't abort, but log + metrics will signal issues)
- Phase 4 training loop (cloud GPU) is out-of-scope for local testing — real training will validate end-to-end
- Wheel-exclusion config is guaranteed; actual build verification deferred to CI/release pipeline

**Status: DONE**

Commit: `824f8ff` (feat: implement data collector scaffold and solver integration)  
Branch: `feat/data-flywheel`
