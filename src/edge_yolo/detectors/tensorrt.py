from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from edge_yolo.coco import COCO_CLASSES
from edge_yolo.cuda_mem import (
    free,
    malloc,
    memcpy_device_to_host,
    memcpy_host_to_device,
    stream_create,
    stream_destroy,
    stream_synchronize,
)
from edge_yolo.deps import require_tensorrt
from edge_yolo.detectors.base import Detector
from edge_yolo.types import OutputLayout

Runner = Callable[[np.ndarray], np.ndarray]


class TensorRtDetector(Detector):
    def __init__(
        self,
        engine_path: Path | None = None,
        runner: Runner | None = None,
        input_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_names: Sequence[str] = COCO_CLASSES,
        layout: OutputLayout = OutputLayout.AUTO,
    ) -> None:
        super().__init__(
            input_size=input_size,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            class_names=class_names,
            layout=layout,
        )
        if runner is not None:
            self._runner = runner
            return
        if engine_path is None:
            raise ValueError("engine_path is required when a TensorRT runner is not supplied")
        engine = load_engine(engine_path)
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError(f"TensorRT could not create an execution context for {engine_path}")
        self._runner = _bind_runner(engine, context)

    @property
    def backend_name(self) -> str:
        return "tensorrt"

    def infer(self, blob: np.ndarray) -> np.ndarray:
        return self._runner(np.ascontiguousarray(blob, dtype=np.float32))


def load_engine(path: Path):
    trt = require_tensorrt()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"failed to deserialize TensorRT engine: {path}")
    return engine


def _bind_runner(engine, context) -> Runner:
    trt = require_tensorrt()
    input_name = None
    output_name = None
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_name = name
        elif mode == trt.TensorIOMode.OUTPUT:
            if output_name is not None:
                raise RuntimeError("this pipeline expects a single TensorRT output")
            output_name = name
    if input_name is None or output_name is None:
        raise RuntimeError("TensorRT engine is missing an input or an output")

    def run(blob: np.ndarray) -> np.ndarray:
        context.set_input_shape(input_name, tuple(blob.shape))
        output_shape = tuple(int(dim) for dim in context.get_tensor_shape(output_name))
        if any(dim < 0 for dim in output_shape):
            raise RuntimeError(f"TensorRT output shape is unresolved: {output_shape}")
        output = np.empty(output_shape, dtype=np.float32)
        device_input = malloc(blob.nbytes)
        device_output = malloc(output.nbytes)
        stream = stream_create()
        try:
            memcpy_host_to_device(device_input, blob)
            context.set_tensor_address(input_name, int(device_input))
            context.set_tensor_address(output_name, int(device_output))
            if not context.execute_async_v3(stream):
                raise RuntimeError("TensorRT execution failed")
            stream_synchronize(stream)
            memcpy_device_to_host(output, device_output)
        finally:
            free(device_input)
            free(device_output)
            stream_destroy(stream)
        return output

    return run
