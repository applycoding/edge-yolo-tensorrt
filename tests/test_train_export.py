from pathlib import Path

import pytest

from edge_yolo.config import TrainConfig, load_export_config, load_train_config
from edge_yolo.export_onnx import export_onnx
from edge_yolo.train import train_from_config

ROOT = Path(__file__).resolve().parents[1]


def test_repo_configs_load():
    train = load_train_config(ROOT / "configs" / "coco128.yaml")
    export = load_export_config(ROOT / "configs" / "export.yaml")
    assert train.model == "yolov8n.pt"
    assert train.data == "coco128.yaml"
    assert train.to_train_kwargs()["device"] is None
    assert train.to_train_kwargs()["fraction"] == 1.0
    assert export.onnx_path.endswith(".onnx")
    assert export.engine_path("fp16").endswith("fp16.engine")
    assert export.engine_path("int8").endswith("int8.engine")
    with pytest.raises(ValueError):
        export.engine_path("fp32")


def test_config_rejects_unknown_keys_and_bad_values(tmp_path: Path):
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text("model: a.pt\ndata: coco128.yaml\nextra: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_train_config(unknown)
    missing = tmp_path / "missing.yaml"
    missing.write_text("epochs: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="model"):
        load_train_config(missing)
    with pytest.raises(ValueError, match="fraction"):
        TrainConfig(model="a.pt", data="coco128.yaml", fraction=0)


def test_train_calls_ultralytics_with_the_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured: dict[str, object] = {}

    class Result:
        save_dir = tmp_path / "run"

    class FakeYOLO:
        def __init__(self, model: str) -> None:
            captured["model"] = model

        def train(self, **kwargs: object) -> Result:
            captured["kwargs"] = kwargs
            return Result()

    monkeypatch.setattr("edge_yolo.deps.YOLO", FakeYOLO)
    summary = train_from_config(
        TrainConfig(model="yolov8n.pt", data="coco128.yaml", epochs=2, device="auto")
    )
    assert captured["model"] == "yolov8n.pt"
    assert captured["kwargs"] == {
        **TrainConfig(model="yolov8n.pt", data="coco128.yaml", epochs=2).to_train_kwargs()
    }
    assert summary.best_weights == tmp_path / "run" / "weights" / "best.pt"


def test_export_onnx_uses_ultralytics_and_moves_the_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    produced = tmp_path / "best.onnx"
    destination = tmp_path / "artifacts" / "model.onnx"

    class FakeYOLO:
        def __init__(self, weights: str) -> None:
            assert weights.endswith("best.pt")

        def export(self, **kwargs: object) -> str:
            assert kwargs["format"] == "onnx"
            assert kwargs["half"] is False
            assert kwargs["imgsz"] == 320
            produced.write_bytes(b"onnx")
            return str(produced)

    monkeypatch.setattr("edge_yolo.deps.YOLO", FakeYOLO)
    exported = export_onnx(tmp_path / "best.pt", destination, imgsz=320, opset=17)
    assert exported == destination
    assert destination.read_bytes() == b"onnx"
    assert not produced.exists()


def test_export_onnx_reports_a_missing_install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr("edge_yolo.deps.YOLO", None)
    with pytest.raises(RuntimeError, match="ultralytics"):
        export_onnx(tmp_path / "best.pt", tmp_path / "model.onnx")
