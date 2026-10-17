from pathlib import Path

import numpy as np
import pytest

from edge_yolo.calibrate import CalibrationBatchSource, CalibrationCache


def test_batch_source_drops_a_partial_tail():
    blobs = [np.full((1, 3, 4, 4), index, dtype=np.float32) for index in range(5)]
    source = CalibrationBatchSource(blobs, batch_size=2)
    assert len(source) == 2
    first = source.next_batch()
    second = source.next_batch()
    assert first is not None and second is not None
    assert first.shape == (2, 3, 4, 4)
    assert second.shape == (2, 3, 4, 4)
    assert float(first[0, 0, 0, 0]) == pytest.approx(0)
    assert float(second[1, 0, 0, 0]) == pytest.approx(3)
    assert source.next_batch() is None
    source.reset()
    replay = source.next_batch()
    assert replay is not None
    assert np.array_equal(replay, first)


def test_batch_source_rejects_short_or_ragged_input():
    one = [np.zeros((3, 4, 4), dtype=np.float32)]
    with pytest.raises(ValueError, match="at least"):
        CalibrationBatchSource(one, batch_size=2)
    ragged = [
        np.zeros((3, 4, 4), dtype=np.float32),
        np.zeros((3, 5, 5), dtype=np.float32),
    ]
    with pytest.raises(ValueError, match="shape"):
        CalibrationBatchSource(ragged, batch_size=1)
    with pytest.raises(ValueError, match="empty"):
        CalibrationBatchSource([], batch_size=1)


def test_calibration_cache_roundtrip(tmp_path: Path):
    cache = CalibrationCache(tmp_path / "nested" / "table.cache")
    assert cache.read() is None
    cache.write(b"table")
    assert cache.read() == b"table"
    cache.path.write_bytes(b"")
    assert cache.read() is None
