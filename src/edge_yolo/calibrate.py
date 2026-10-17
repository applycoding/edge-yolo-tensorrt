from pathlib import Path

import numpy as np

from edge_yolo.cuda_mem import DeviceBuffer
from edge_yolo.deps import require_tensorrt
from edge_yolo.preprocess import iter_image_paths, load_rgb_image, prepare_input


class CalibrationCache:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> bytes | None:
        if not self.path.is_file():
            return None
        payload = self.path.read_bytes()
        if not payload:
            return None
        return payload

    def write(self, payload: bytes) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(payload)


class CalibrationBatchSource:
    """Yields NCHW float batches. A trailing partial batch is dropped."""

    def __init__(self, blobs: list[np.ndarray], batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not blobs:
            raise ValueError("calibration set is empty")
        images = [_single_image(blob) for blob in blobs]
        reference = images[0].shape
        for image in images:
            if image.shape != reference:
                raise ValueError("calibration images must share one shape after preprocess")
        usable = (len(images) // batch_size) * batch_size
        if usable == 0:
            raise ValueError(f"need at least {batch_size} calibration images, found {len(images)}")
        self._images = images[:usable]
        self._batch_size = batch_size
        self._cursor = 0

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def __len__(self) -> int:
        return len(self._images) // self._batch_size

    def reset(self) -> None:
        self._cursor = 0

    def next_batch(self) -> np.ndarray | None:
        if self._cursor >= len(self._images):
            return None
        end = self._cursor + self._batch_size
        batch = np.stack(self._images[self._cursor : end], axis=0)
        self._cursor = end
        return np.ascontiguousarray(batch, dtype=np.float32)


def load_calibration_blobs(directory: Path, input_size: int) -> list[np.ndarray]:
    blobs: list[np.ndarray] = []
    for path in iter_image_paths(directory):
        image = load_rgb_image(path)
        blob, _meta = prepare_input(image, input_size)
        blobs.append(blob)
    return blobs


def make_entropy_calibrator(source: CalibrationBatchSource, cache: CalibrationCache):
    """TensorRT entropy calibrator. The returned object is only valid while TensorRT is imported."""
    runtime = require_tensorrt()

    class EntropyCalibrator(runtime.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            runtime.IInt8EntropyCalibrator2.__init__(self)
            self._source = source
            self._cache = cache
            self._device = DeviceBuffer()

        def get_batch_size(self) -> int:
            return self._source.batch_size

        def get_batch(self, names: list[str]) -> list[int] | None:
            del names
            batch = self._source.next_batch()
            if batch is None:
                return None
            return [self._device.upload(batch)]

        def read_calibration_cache(self) -> bytes | None:
            return self._cache.read()

        def write_calibration_cache(self, payload: bytes) -> None:
            self._cache.write(bytes(payload))

    return EntropyCalibrator()


def _single_image(blob: np.ndarray) -> np.ndarray:
    array = np.asarray(blob, dtype=np.float32)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError("calibration blobs must be a single image")
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"calibration blob must be CHW or NCHW, got shape {array.shape}")
    return np.ascontiguousarray(array)
