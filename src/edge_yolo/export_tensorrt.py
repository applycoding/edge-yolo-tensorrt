from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from edge_yolo.calibrate import (
    CalibrationBatchSource,
    CalibrationCache,
    load_calibration_blobs,
    make_entropy_calibrator,
)
from edge_yolo.deps import require_tensorrt
from edge_yolo.types import Precision


@dataclass(frozen=True, slots=True)
class EngineBuildPlan:
    precision: Precision
    enable_fp16: bool
    enable_int8: bool
    requires_calibrator: bool


def plan_engine_build(precision: Precision) -> EngineBuildPlan:
    """INT8 builds also enable FP16 so layers without an INT8 kernel can fall back."""
    match precision:
        case Precision.FP32:
            return EngineBuildPlan(
                precision,
                enable_fp16=False,
                enable_int8=False,
                requires_calibrator=False,
            )
        case Precision.FP16:
            return EngineBuildPlan(
                precision,
                enable_fp16=True,
                enable_int8=False,
                requires_calibrator=False,
            )
        case Precision.INT8:
            return EngineBuildPlan(
                precision,
                enable_fp16=True,
                enable_int8=True,
                requires_calibrator=True,
            )
        case _:
            assert_never(precision)


def validate_static_batch(onnx_batch: int | None, batch_size: int) -> None:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if onnx_batch not in (None, -1) and onnx_batch != batch_size:
        raise ValueError(
            f"ONNX input batch is {onnx_batch}, but the engine batch size is {batch_size}"
        )


def build_engine(
    onnx_path: Path,
    output_path: Path,
    precision: Precision,
    imgsz: int = 640,
    workspace_mib: int = 2048,
    batch_size: int = 1,
    calibrator: object | None = None,
) -> Path:
    plan = plan_engine_build(precision)
    if plan.requires_calibrator and calibrator is None:
        raise ValueError(
            "INT8 export requires a calibrator. Pass a calibration directory or an existing cache."
        )
    if imgsz < 32:
        raise ValueError("imgsz must be >= 32")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if workspace_mib < 1:
        raise ValueError("workspace_mib must be >= 1")

    trt = require_tensorrt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        messages = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        details = "\n".join(messages)
        raise RuntimeError(f"failed to parse ONNX:\n{details}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib * 1024 * 1024)
    if plan.enable_fp16:
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("FP16 was requested, but this GPU does not advertise fast FP16")
        config.set_flag(trt.BuilderFlag.FP16)
    if plan.enable_int8:
        if not builder.platform_has_fast_int8:
            raise RuntimeError("INT8 was requested, but this GPU does not advertise fast INT8")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator

    input_tensor = network.get_input(0)
    if input_tensor is None:
        raise RuntimeError("ONNX network has no inputs")
    shape = tuple(input_tensor.shape)
    onnx_batch = int(shape[0]) if shape else None
    validate_static_batch(onnx_batch, batch_size)
    if any(dim < 0 for dim in shape):
        profile = builder.create_optimization_profile()
        fixed = (batch_size, 3, imgsz, imgsz)
        profile.set_shape(input_tensor.name, fixed, fixed, fixed)
        config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT did not return an engine")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(memoryview(serialized).tobytes())
    return output_path


def build_engine_from_images(
    onnx_path: Path,
    output_path: Path,
    precision: Precision,
    calibration_dir: Path | None,
    calibration_cache: Path,
    imgsz: int = 640,
    workspace_mib: int = 2048,
    batch_size: int = 1,
) -> Path:
    plan = plan_engine_build(precision)
    calibrator = None
    if plan.requires_calibrator:
        if calibration_dir is None:
            raise ValueError("INT8 export requires --calibration-dir")
        cache = CalibrationCache(calibration_cache)
        blobs = load_calibration_blobs(calibration_dir, imgsz)
        source = CalibrationBatchSource(blobs, batch_size)
        calibrator = make_entropy_calibrator(source, cache)
    return build_engine(
        onnx_path=onnx_path,
        output_path=output_path,
        precision=precision,
        imgsz=imgsz,
        workspace_mib=workspace_mib,
        batch_size=batch_size,
        calibrator=calibrator,
    )
