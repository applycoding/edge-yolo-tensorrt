from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class Precision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


class Backend(Enum):
    ONNX = "onnx"
    TENSORRT = "tensorrt"


class OutputLayout(Enum):
    AUTO = "auto"
    YOLOV8 = "yolov8"
    YOLOV5 = "yolov5"
    END2END = "end2end"


@dataclass(frozen=True, slots=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
