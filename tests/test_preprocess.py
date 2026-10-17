from pathlib import Path

import numpy as np
import pytest

from edge_yolo.preprocess import letterbox, load_rgb_image, prepare_input


def test_letterbox_pads_the_short_side():
    image = np.full((2, 4, 3), 255, dtype=np.uint8)
    canvas, meta = letterbox(image, 4)
    assert canvas.shape == (4, 4, 3)
    assert meta.scale == pytest.approx(1.0)
    assert meta.pad_x == 0
    assert meta.pad_y == 1
    assert np.all(canvas[0] == 114)
    assert np.all(canvas[3] == 114)
    assert np.all(canvas[1:3] == 255)


def test_prepare_input_is_nchw_unit_float():
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    blob, meta = prepare_input(image, 32)
    assert blob.shape == (1, 3, 32, 32)
    assert blob.dtype == np.float32
    assert blob.flags.c_contiguous
    assert float(blob.max()) == pytest.approx(1.0)
    assert meta.original_width == 32
    assert meta.pad_x == 0


def test_prepare_input_rejects_non_rgb_uint8():
    with pytest.raises(ValueError):
        prepare_input(np.zeros((8, 8), dtype=np.uint8), 32)
    with pytest.raises(ValueError):
        prepare_input(np.zeros((32, 32, 3), dtype=np.float32), 32)


def test_load_npy_rgb(tmp_path: Path):
    path = tmp_path / "frame.npy"
    image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    np.save(path, image)
    loaded = load_rgb_image(path)
    assert np.array_equal(loaded, image)
