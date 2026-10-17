import json
from pathlib import Path

import numpy as np
import pytest

from edge_yolo.cli import main
from edge_yolo.train import TrainSummary
from edge_yolo.types import Detection


def test_help_exits_cleanly():
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0


def test_train_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    config = tmp_path / "train.yaml"
    config.write_text("model: yolov8n.pt\ndata: coco128.yaml\nepochs: 1\n", encoding="utf-8")
    seen: dict[str, int] = {}

    def fake_train(loaded):
        seen["epochs"] = loaded.epochs
        return TrainSummary(tmp_path, tmp_path / "best.pt")

    monkeypatch.setattr("edge_yolo.cli.train_from_config", fake_train)
    assert main(["train", "--config", str(config)]) == 0
    assert seen["epochs"] == 1


def test_export_commands(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    def fake_onnx(weights: Path, output: Path, **kwargs: object) -> Path:
        assert kwargs["opset"] == 13
        assert weights.name == "a.pt"
        return output

    monkeypatch.setattr("edge_yolo.cli.export_onnx", fake_onnx)
    assert main(["export-onnx", "--weights", "a.pt", "--output", "a.onnx", "--opset", "13"]) == 0

    code = main(["export-trt", "--onnx", "m.onnx", "--output", "m.engine", "--precision", "int8"])
    assert code == 1
    assert "calibration" in capsys.readouterr().err


def test_infer_and_bench(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    image_path = tmp_path / "frame.npy"
    np.save(image_path, np.zeros((8, 8, 3), dtype=np.uint8))
    detection = Detection(1, 2, 3, 4, 0.5, 0, "person")

    class FakeDetector:
        backend_name = "onnx"

        def predict(self, image: np.ndarray) -> list[Detection]:
            assert image.shape[-1] == 3
            return [detection]

    monkeypatch.setattr("edge_yolo.cli.open_detector", lambda *_args, **_kwargs: FakeDetector())
    assert (
        main(
            [
                "infer",
                "--backend",
                "onnx",
                "--model",
                "model.onnx",
                "--image",
                str(image_path),
                "--input-size",
                "32",
            ]
        )
        == 0
    )
    assert "person" in capsys.readouterr().out

    report_path = tmp_path / "report.json"
    assert (
        main(
            [
                "bench",
                "--backend",
                "onnx",
                "--model",
                "model.onnx",
                "--input-size",
                "32",
                "--precision",
                "fp16",
                "--warmup",
                "1",
                "--iterations",
                "2",
                "--height",
                "8",
                "--width",
                "8",
                "--images-count",
                "1",
                "--device",
                "unit-test",
                "--json",
                str(report_path),
            ]
        )
        == 0
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["device"] == "unit-test"
    assert payload["backend"] == "onnx"
    assert payload["precision"] == "fp16"
    assert payload["iterations"] == 2
    assert isinstance(payload["fps"], float)
