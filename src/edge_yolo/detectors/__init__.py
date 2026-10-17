from pathlib import Path
from typing import assert_never

from edge_yolo.detectors.base import Detector
from edge_yolo.detectors.onnx_runtime import OnnxRuntimeDetector
from edge_yolo.detectors.tensorrt import TensorRtDetector
from edge_yolo.types import Backend, OutputLayout


def open_detector(
    backend: Backend,
    model_path: Path,
    *,
    input_size: int = 640,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    layout: OutputLayout = OutputLayout.AUTO,
    providers: list[str] | None = None,
) -> Detector:
    match backend:
        case Backend.ONNX:
            return OnnxRuntimeDetector(
                model_path,
                providers=providers,
                input_size=input_size,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                layout=layout,
            )
        case Backend.TENSORRT:
            if providers is not None:
                raise ValueError("providers is only used by the ONNX backend")
            return TensorRtDetector(
                model_path,
                input_size=input_size,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                layout=layout,
            )
        case _:
            assert_never(backend)


__all__ = [
    "Backend",
    "Detector",
    "OnnxRuntimeDetector",
    "TensorRtDetector",
    "open_detector",
]
