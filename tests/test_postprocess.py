import numpy as np
import pytest

from edge_yolo.postprocess import decode_output, nms, scale_boxes, xywh_to_xyxy
from edge_yolo.preprocess import LetterboxMeta
from edge_yolo.types import OutputLayout


def test_xywh_to_xyxy():
    converted = xywh_to_xyxy(np.array([[16, 16, 8, 8]], dtype=np.float32))
    assert converted[0].tolist() == pytest.approx([12, 12, 20, 20])


def test_nms_suppresses_overlaps_and_keeps_separated_boxes():
    boxes = np.array(
        [
            [0, 0, 10, 10],
            [1, 1, 11, 11],
            [30, 30, 40, 40],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    assert nms(boxes, scores, 0.5) == [0, 2]


def test_yolov8_keeps_the_high_score_anchor():
    prediction = _yolov8([(16, 16, 8, 8, 0, 0.9), (4, 4, 2, 2, 0, 0.05)])
    boxes, scores, class_ids = decode_output(prediction, OutputLayout.YOLOV8, 0.25, 0.5)
    assert len(scores) == 1
    assert scores[0] == pytest.approx(0.9)
    assert class_ids[0] == 0
    assert boxes[0].tolist() == pytest.approx([12, 12, 20, 20])


def test_auto_layout_accepts_channel_first_yolov8_and_end2end():
    yolov8 = _yolov8([(16, 16, 8, 8, 0, 0.8)])
    end2end = np.array([[[8, 12, 16, 20, 0.91, 2], [0, 0, 1, 1, 0.01, 0]]], dtype=np.float32)
    yolo_boxes, yolo_scores, _ = decode_output(yolov8, OutputLayout.AUTO, 0.25, 0.5)
    end_boxes, end_scores, end_classes = decode_output(end2end, OutputLayout.AUTO, 0.25, 0.5)
    assert len(yolo_scores) == 1
    assert yolo_boxes[0, 0] == pytest.approx(12)
    assert len(end_scores) == 1
    assert end_scores[0] == pytest.approx(0.91)
    assert end_classes[0] == 2
    assert end_boxes[0].tolist() == pytest.approx([8, 12, 16, 20])


def test_class_aware_nms_keeps_different_classes():
    rows = np.array(
        [
            [0, 0, 10, 10, 0.9, 0],
            [1, 1, 11, 11, 0.8, 0],
            [30, 30, 40, 40, 0.7, 0],
            [1, 1, 11, 11, 0.6, 1],
        ],
        dtype=np.float32,
    )
    _boxes, scores, class_ids = decode_output(rows[None], OutputLayout.END2END, 0.25, 0.5)
    assert scores.tolist() == pytest.approx([0.9, 0.7, 0.6])
    assert class_ids.tolist() == [0, 0, 1]


def test_yolov5_multiplies_objectness():
    prediction = np.zeros((1, 6, 8), dtype=np.float32)
    prediction[0, :, 0] = np.array([16, 16, 8, 8, 0.4, 0.5])
    prediction[0, :, 1] = np.array([8, 8, 4, 4, 1.0, 0.9])
    _boxes, scores, class_ids = decode_output(prediction, OutputLayout.YOLOV5, 0.25, 0.5)
    assert len(scores) == 1
    assert scores[0] == pytest.approx(0.9)
    assert class_ids[0] == 0


def test_empty_prediction_and_bad_batch():
    empty = np.zeros((1, 5, 8), dtype=np.float32)
    boxes, scores, class_ids = decode_output(empty, OutputLayout.YOLOV8, 0.25, 0.5)
    assert boxes.shape == (0, 4)
    assert scores.shape == (0,)
    assert class_ids.shape == (0,)
    with pytest.raises(ValueError):
        decode_output(np.zeros((2, 5, 8), dtype=np.float32), OutputLayout.YOLOV8, 0.25, 0.5)


def test_scale_boxes_undoes_letterbox():
    boxes = np.array([[8, 12, 16, 20]], dtype=np.float32)
    meta = LetterboxMeta(
        scale=0.5,
        pad_x=0,
        pad_y=8,
        original_width=64,
        original_height=32,
        input_size=32,
    )
    scaled = scale_boxes(boxes, meta)
    assert scaled[0].tolist() == pytest.approx([16, 8, 32, 24])


def _yolov8(records: list[tuple[float, float, float, float, int, float]]) -> np.ndarray:
    num_classes = 1
    anchors = max(8, len(records) + num_classes + 4)
    prediction = np.zeros((1, 4 + num_classes, anchors), dtype=np.float32)
    for index, (cx, cy, width, height, class_id, score) in enumerate(records):
        prediction[0, 0, index] = cx
        prediction[0, 1, index] = cy
        prediction[0, 2, index] = width
        prediction[0, 3, index] = height
        prediction[0, 4 + class_id, index] = score
    return prediction
