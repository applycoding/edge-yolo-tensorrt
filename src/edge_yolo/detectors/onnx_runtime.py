from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from edge_yolo.coco import COCO_CLASSES
from edge_yolo.deps import require_onnxruntime
from edge_yolo.detectors.base import Detector
from edge_yolo.types import OutputLayout


class _ModelInput(Protocol):
    name: str


class _Session(Protocol):
    def get_inputs(self) -> Sequence[_ModelInput]: ...

    def run(self, output_names: object, feeds: dict[str, np.ndarray]) -> Sequence[np.ndarray]: ...


class OnnxRuntimeDetector(Detector):
    def __init__(
        self,
        model_path: Path | None = None,
        session: _Session | None = None,
        providers: Sequence[str] | None = None,
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
        self._session = session if session is not None else _open_session(model_path, providers)
        inputs = self._session.get_inputs()
        if not inputs:
            raise RuntimeError("ONNX session has no inputs")
        self._input_name = inputs[0].name

    @property
    def backend_name(self) -> str:
        return "onnx"

    def infer(self, blob: np.ndarray) -> np.ndarray:
        outputs = self._session.run(None, {self._input_name: blob})
        if not outputs:
            raise RuntimeError("ONNX Runtime returned no outputs")
        return np.asarray(outputs[0])


def _open_session(model_path: Path | None, providers: Sequence[str] | None) -> _Session:
    if model_path is None:
        raise ValueError("model_path is required when an ONNX session is not supplied")
    runtime = require_onnxruntime()
    options = runtime.SessionOptions()
    options.graph_optimization_level = runtime.GraphOptimizationLevel.ORT_ENABLE_ALL
    if providers is None:
        return runtime.InferenceSession(str(model_path), sess_options=options)
    return runtime.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=list(providers),
    )
