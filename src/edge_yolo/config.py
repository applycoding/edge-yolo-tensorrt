from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    model: str
    data: str
    epochs: int = 20
    imgsz: int = 640
    batch: int = 8
    device: str = "auto"
    workers: int = 2
    patience: int = 10
    optimizer: str = "auto"
    lr0: float = 0.01
    project: str = "runs/detect"
    name: str = "coco128"
    exist_ok: bool = True
    pretrained: bool = True
    seed: int = 0
    fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.imgsz < 32:
            raise ValueError("imgsz must be >= 32")
        if self.batch < 1:
            raise ValueError("batch must be >= 1")
        if not 0 < self.fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        if self.workers < 0:
            raise ValueError("workers must be >= 0")

    def to_train_kwargs(self) -> dict[str, Any]:
        device = None if self.device == "auto" else self.device
        return {
            "data": self.data,
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "device": device,
            "workers": self.workers,
            "patience": self.patience,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "project": self.project,
            "name": self.name,
            "exist_ok": self.exist_ok,
            "pretrained": self.pretrained,
            "seed": self.seed,
            "fraction": self.fraction,
        }


@dataclass
class ExportConfig:
    weights: str
    imgsz: int = 640
    opset: int = 17
    simplify: bool = True
    dynamic: bool = False
    onnx_path: str = "artifacts/model.onnx"
    fp16_engine: str = "artifacts/model_fp16.engine"
    int8_engine: str = "artifacts/model_int8.engine"
    workspace_mib: int = 2048
    calibration_dir: str = "data/calibration"
    calibration_cache: str = "artifacts/int8.cache"

    def __post_init__(self) -> None:
        if self.imgsz < 32:
            raise ValueError("imgsz must be >= 32")
        if self.opset < 12:
            raise ValueError("opset must be >= 12")
        if self.workspace_mib < 1:
            raise ValueError("workspace_mib must be >= 1")

    def engine_path(self, precision: str) -> str:
        if precision == "int8":
            return self.int8_engine
        if precision == "fp16":
            return self.fp16_engine
        raise ValueError(f"no engine path configured for precision {precision!r}")


def load_train_config(path: Path) -> TrainConfig:
    data = _load_mapping(path)
    _reject_unknown(data, TrainConfig, path)
    for key in ("model", "data"):
        if key not in data:
            raise ValueError(f"{path} is missing required key {key!r}")
    return TrainConfig(**data)


def load_export_config(path: Path) -> ExportConfig:
    data = _load_mapping(path)
    _reject_unknown(data, ExportConfig, path)
    if "weights" not in data:
        raise ValueError(f"{path} is missing required key 'weights'")
    return ExportConfig(**data)


def _load_mapping(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


def _reject_unknown(data: dict[str, Any], cls: type, path: Path) -> None:
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"{path} has unknown keys: {joined}")
