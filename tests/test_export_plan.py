from pathlib import Path

import numpy as np
import pytest

from edge_yolo.export_tensorrt import (
    build_engine,
    build_engine_from_images,
    plan_engine_build,
    validate_static_batch,
)
from edge_yolo.types import Precision


def test_plan_covers_every_precision():
    plans = {precision: plan_engine_build(precision) for precision in Precision}
    assert plans[Precision.FP32].enable_fp16 is False
    assert plans[Precision.FP32].requires_calibrator is False
    assert plans[Precision.FP16].enable_fp16 is True
    assert plans[Precision.FP16].enable_int8 is False
    assert plans[Precision.INT8].enable_int8 is True
    assert plans[Precision.INT8].enable_fp16 is True
    assert plans[Precision.INT8].requires_calibrator is True


def test_static_batch_must_match_a_fixed_onnx_batch():
    validate_static_batch(1, 1)
    validate_static_batch(-1, 4)
    validate_static_batch(None, 2)
    with pytest.raises(ValueError, match="batch"):
        validate_static_batch(1, 4)


def test_int8_build_requires_a_calibrator(tmp_path: Path):
    with pytest.raises(ValueError, match="calibrator"):
        build_engine(
            tmp_path / "model.onnx",
            tmp_path / "model.engine",
            Precision.INT8,
            imgsz=32,
        )


def test_fp16_build_requires_tensorrt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr("edge_yolo.deps.trt", None)
    with pytest.raises(RuntimeError, match="tensorrt"):
        build_engine(
            tmp_path / "model.onnx",
            tmp_path / "model.engine",
            Precision.FP16,
            imgsz=32,
        )


def test_int8_from_images_requires_tensorrt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr("edge_yolo.deps.trt", None)
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    np.save(calibration / "frame.npy", np.zeros((32, 32, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="tensorrt"):
        build_engine_from_images(
            tmp_path / "model.onnx",
            tmp_path / "model.engine",
            Precision.INT8,
            calibration,
            tmp_path / "table.cache",
            imgsz=32,
        )


def test_int8_from_images_requires_a_directory(tmp_path: Path):
    with pytest.raises(ValueError, match="calibration"):
        build_engine_from_images(
            tmp_path / "model.onnx",
            tmp_path / "model.engine",
            Precision.INT8,
            None,
            tmp_path / "table.cache",
            imgsz=32,
        )
