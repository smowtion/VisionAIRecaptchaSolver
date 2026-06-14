"""Resolve the best available training device for ultralytics (CUDA > MPS > CPU).

Lets the training scripts run unchanged on a CUDA GPU (cloud), an Apple Silicon Mac
(MPS), or CPU. Importing torch is deferred so the module imports without torch present.
"""

from __future__ import annotations


def resolve_device(device: str | int | None = None) -> int | str:
    """Resolve a device for ``YOLO.train(device=...)``.

    Args:
        device: Explicit device ("0", "mps", "cpu", an int index), or None/"auto" to
            auto-detect.

    Returns:
        An explicit device: CUDA index 0 if CUDA is available, else "mps" on Apple
        Silicon, else "cpu". An explicit (non-"auto") value is returned as-is (int if a
        digit string).
    """
    if device is not None and str(device).lower() != "auto":
        d = str(device)
        return int(d) if d.isdigit() else d

    try:
        import torch

        if torch.cuda.is_available():
            return 0
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # torch missing / probe failed -> safe CPU fallback
        pass
    return "cpu"
