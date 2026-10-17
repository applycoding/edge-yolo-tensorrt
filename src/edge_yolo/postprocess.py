from typing import assert_never

import numpy as np

from edge_yolo.preprocess import LetterboxMeta
from edge_yolo.types import OutputLayout

# A short (N, 6) tensor is end-to-end NMS output. Longer tensors are raw heads.
_END2END_MAX_ROWS = 1000


def decode_output(
    output: np.ndarray,
    layout: OutputLayout,
    conf_threshold: float,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction = _squeeze_batch(output)
    resolved = resolve_layout(prediction, layout)
    match resolved:
        case OutputLayout.YOLOV8:
            return _decode_raw(prediction, conf_threshold, iou_threshold, objectness=False)
        case OutputLayout.YOLOV5:
            return _decode_raw(prediction, conf_threshold, iou_threshold, objectness=True)
        case OutputLayout.END2END:
            return _decode_end2end(prediction, conf_threshold, iou_threshold)
        case OutputLayout.AUTO:
            raise RuntimeError("AUTO layout was not resolved")
        case _:
            assert_never(resolved)


def resolve_layout(prediction: np.ndarray, requested: OutputLayout) -> OutputLayout:
    match requested:
        case OutputLayout.AUTO:
            return _infer_layout(prediction)
        case OutputLayout.YOLOV8 | OutputLayout.YOLOV5 | OutputLayout.END2END:
            return requested
        case _:
            assert_never(requested)


def scale_boxes(boxes: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    if len(boxes) == 0:
        return boxes.astype(np.float32, copy=False)
    scaled = boxes.astype(np.float32, copy=True)
    scaled[:, [0, 2]] -= meta.pad_x
    scaled[:, [1, 3]] -= meta.pad_y
    scaled /= np.float32(meta.scale)
    scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0, meta.original_width)
    scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0, meta.original_height)
    return scaled


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        overlap = _iou_vector(boxes[current], boxes[rest])
        order = rest[overlap <= iou_threshold]
    return keep


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = np.empty_like(boxes, dtype=np.float32)
    half_width = boxes[:, 2] / 2
    half_height = boxes[:, 3] / 2
    converted[:, 0] = boxes[:, 0] - half_width
    converted[:, 1] = boxes[:, 1] - half_height
    converted[:, 2] = boxes[:, 0] + half_width
    converted[:, 3] = boxes[:, 1] + half_height
    return converted


def _decode_raw(
    prediction: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    objectness: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _as_anchor_rows(prediction)
    box_columns = 5 if objectness else 4
    if rows.shape[1] <= box_columns:
        raise ValueError(
            f"raw YOLO output needs more than {box_columns} channels, got {rows.shape[1]}"
        )
    boxes = xywh_to_xyxy(rows[:, :4])
    class_scores = rows[:, box_columns:]
    class_ids = class_scores.argmax(axis=1).astype(np.int64)
    scores = class_scores[np.arange(len(rows)), class_ids]
    if objectness:
        scores = scores * rows[:, 4]
    return _filter_and_nms(
        boxes,
        scores.astype(np.float32),
        class_ids,
        conf_threshold,
        iou_threshold,
    )


def _decode_end2end(
    prediction: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = prediction.T if prediction.shape[0] == 6 and prediction.shape[1] != 6 else prediction
    if rows.shape[1] != 6:
        raise ValueError(f"end-to-end output must have 6 columns, got shape {prediction.shape}")
    boxes = rows[:, :4].astype(np.float32)
    scores = rows[:, 4].astype(np.float32)
    class_ids = np.rint(rows[:, 5]).astype(np.int64)
    return _filter_and_nms(boxes, scores, class_ids, conf_threshold, iou_threshold)


def _filter_and_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(scores) & np.isfinite(boxes).all(axis=1)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    keep_mask = finite & (scores >= conf_threshold) & (widths > 0) & (heights > 0)
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    class_ids = class_ids[keep_mask]
    if len(boxes) == 0:
        return _empty()

    selected: list[int] = []
    for class_id in np.unique(class_ids):
        indexes = np.flatnonzero(class_ids == class_id)
        for local in nms(boxes[indexes], scores[indexes], iou_threshold):
            selected.append(int(indexes[local]))
    if not selected:
        return _empty()
    order = np.array(selected, dtype=np.int64)
    order = order[np.argsort(scores[order])[::-1]]
    return boxes[order], scores[order], class_ids[order]


def _infer_layout(prediction: np.ndarray) -> OutputLayout:
    rows, columns = prediction.shape
    if columns == 6 and rows <= _END2END_MAX_ROWS:
        return OutputLayout.END2END
    if rows == 6 and columns <= _END2END_MAX_ROWS and rows > columns:
        return OutputLayout.END2END
    return OutputLayout.YOLOV8


def _squeeze_batch(output: np.ndarray) -> np.ndarray:
    array = np.asarray(output)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError("only batch size 1 predictions are supported")
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"expected a 2D prediction, got shape {array.shape}")
    return array


def _as_anchor_rows(prediction: np.ndarray) -> np.ndarray:
    if prediction.shape[0] < prediction.shape[1]:
        rows = np.ascontiguousarray(prediction.T)
    else:
        rows = prediction
    if rows.shape[1] < 5:
        raise ValueError(
            f"prediction has {rows.shape[1]} channels; expected box values plus class scores"
        )
    return rows


def _iou_vector(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area + areas - intersection, 1e-9)


def _empty() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
    )
