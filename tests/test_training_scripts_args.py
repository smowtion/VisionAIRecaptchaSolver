"""Phase 4 dry tests: training scripts import + parse args without a GPU / real training."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest


def test_train_module_imports_and_has_defaults() -> None:
    import train

    sig = inspect.signature(train.train)
    assert sig.parameters["epochs"].default == 50
    assert sig.parameters["imgsz"].default == 640
    assert sig.parameters["batch"].default == 64
    assert sig.parameters["amp"].default is True
    assert sig.parameters["device"].default is None  # auto-detect


def test_resolve_device_explicit_passthrough() -> None:
    from device_utils import resolve_device

    assert resolve_device("0") == 0
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("mps") == "mps"
    # auto-detect returns one of the valid backends for this machine
    assert resolve_device() in (0, "mps", "cpu")
    assert resolve_device("auto") in (0, "mps", "cpu")


def test_train_cli_help() -> None:
    import train
    from click.testing import CliRunner

    result = CliRunner().invoke(train.main, ["--help"])
    assert result.exit_code == 0
    assert "epochs" in result.output


def test_export_rejects_missing_weights() -> None:
    import export_onnx

    with pytest.raises(FileNotFoundError):
        export_onnx.export(Path("does-not-exist.pt"))


def test_export_cli_help() -> None:
    import export_onnx
    from click.testing import CliRunner

    result = CliRunner().invoke(export_onnx.main, ["--help"])
    assert result.exit_code == 0
    assert "weights" in result.output.lower()


def test_compute_sha256_matches_hashlib(tmp_path: Path) -> None:
    import compute_sha256

    f = tmp_path / "model.onnx"
    f.write_bytes(b"hello-recaptcha-model")

    digest = compute_sha256.compute_sha256(f)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)
    assert digest == hashlib.sha256(b"hello-recaptcha-model").hexdigest()


def test_compute_sha256_missing_file_raises(tmp_path: Path) -> None:
    import compute_sha256

    with pytest.raises(FileNotFoundError):
        compute_sha256.compute_sha256(tmp_path / "nope.onnx")
