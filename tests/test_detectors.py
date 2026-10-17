from pathlib import Path

import numpy as np
import pytest

from edge_yolo.detectors import open_detector
from edge_yolo.detectors.base import Detector
from edge_yolo.detectors.onnx_runtime import OnnxRuntimeDetector
from edge_yolo.detectors.tensorrt import TensorRtDetector
from edge_yolo.types import Backend, OutputLayout


class _Input:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeSession:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.feeds: list[dict[str, np.ndarray]] = []

    def get_inputs(self) -> list[_Input]:
        return [_Input("images")]

    def run(self, _names: object, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.feeds.append(feeds)
        return [self.output]


class ScriptedDetector(Detector):
    def __init__(self, output: np.ndarray, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.output = output
        self.last_blob: np.ndarray | None = None

    @property
    def backend_name(self) -> str:
        return "scripted"

    def infer(self, blob: np.ndarray) -> np.ndarray:
        self.last_blob = blob
        return self.output


def test_shared_predict_scales_boxes_and_names_classes():
    output = np.array([[[8, 12, 16, 20, 0.9, 0]]], dtype=np.float32)
    detector = ScriptedDetector(
        output,
        input_size=32,
        class_names=("person", "bicycle"),
        layout=OutputLayout.END2END,
    )
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    detections = detector.predict(image)
    assert detector.last_blob is not None
    assert detector.last_blob.shape == (1, 3, 32, 32)
    assert len(detections) == 1
    found = detections[0]
    assert found.class_name == "person"
    assert found.confidence == pytest.approx(0.9)
    assert [found.x1, found.y1, found.x2, found.y2] == pytest.approx([16, 8, 32, 24])


def test_onnx_and_tensorrt_share_the_decoder():
    output = np.array([[[4, 4, 12, 12, 0.7, 1]]], dtype=np.float32)
    session = FakeSession(output)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    onnx = OnnxRuntimeDetector(
        session=session,
        input_size=32,
        class_names=("person", "bicycle"),
        layout=OutputLayout.END2END,
    )
    tensorrt = TensorRtDetector(
        runner=lambda _blob: output,
        input_size=32,
        class_names=("person", "bicycle"),
        layout=OutputLayout.END2END,
    )
    onnx_hits = onnx.predict(image)
    trt_hits = tensorrt.predict(image)
    assert onnx.backend_name == "onnx"
    assert tensorrt.backend_name == "tensorrt"
    assert onnx_hits[0].class_name == trt_hits[0].class_name == "bicycle"
    assert onnx_hits[0].to_dict()["x1"] == pytest.approx(trt_hits[0].x1)
    assert session.feeds[0]["images"].shape == (1, 3, 32, 32)


def test_open_detector_stays_offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr("edge_yolo.deps.ort", None)
    monkeypatch.setattr("edge_yolo.deps.trt", None)
    with pytest.raises(RuntimeError, match="onnxruntime"):
        open_detector(Backend.ONNX, tmp_path / "model.onnx")
    with pytest.raises(RuntimeError, match="tensorrt"):
        open_detector(Backend.TENSORRT, tmp_path / "model.engine")
    with pytest.raises(ValueError, match="providers"):
        open_detector(
            Backend.TENSORRT,
            tmp_path / "model.engine",
            providers=["CPUExecutionProvider"],
        )


def test_detector_rejects_bad_thresholds():
    with pytest.raises(ValueError):
        ScriptedDetector(np.zeros((1, 1, 6)), input_size=32, conf_threshold=1.5)
