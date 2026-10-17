from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from edge_yolo.coco import COCO_CLASSES, class_name
from edge_yolo.postprocess import decode_output, scale_boxes
from edge_yolo.preprocess import prepare_input
from edge_yolo.types import Detection, OutputLayout


class Detector(ABC):
    """Shared preprocess, decode, and NMS around a backend-specific `infer` call."""

    def __init__(
        self,
        input_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_names: Sequence[str] = COCO_CLASSES,
        layout: OutputLayout = OutputLayout.AUTO,
    ) -> None:
        if input_size < 32:
            raise ValueError("input_size must be >= 32")
        if not 0 <= conf_threshold <= 1:
            raise ValueError("conf_threshold must be in [0, 1]")
        if not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold must be in [0, 1]")
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = tuple(class_names)
        self.layout = layout

    @property
    @abstractmethod
    def backend_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def infer(self, blob: np.ndarray) -> np.ndarray:
        """Run the network. `blob` is NCHW float32 in 0..1."""
        raise NotImplementedError

    def predict(self, image: np.ndarray) -> list[Detection]:
        blob, meta = prepare_input(image, self.input_size)
        raw = self.infer(blob)
        boxes, scores, class_ids = decode_output(
            raw,
            self.layout,
            self.conf_threshold,
            self.iou_threshold,
        )
        boxes = scale_boxes(boxes, meta)
        detections: list[Detection] = []
        for index in range(len(scores)):
            class_id = int(class_ids[index])
            detections.append(
                Detection(
                    x1=float(boxes[index, 0]),
                    y1=float(boxes[index, 1]),
                    x2=float(boxes[index, 2]),
                    y2=float(boxes[index, 3]),
                    confidence=float(scores[index]),
                    class_id=class_id,
                    class_name=class_name(class_id, self.class_names),
                )
            )
        return detections

    def predict_many(self, images: Sequence[np.ndarray]) -> list[list[Detection]]:
        return [self.predict(image) for image in images]
