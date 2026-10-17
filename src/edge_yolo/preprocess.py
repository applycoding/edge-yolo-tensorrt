from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edge_yolo.deps import Image, cv2

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True, slots=True)
class LetterboxMeta:
    scale: float
    pad_x: int
    pad_y: int
    original_width: int
    original_height: int
    input_size: int


def load_rgb_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
        _require_hwc_uint8(array, str(path))
        return array
    if Image is None:
        raise RuntimeError(
            "Pillow is required to read image files. "
            "Install the media extra, or pass a .npy uint8 HWC array."
        )
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.uint8)
    _require_hwc_uint8(array, str(path))
    return array


def prepare_input(image: np.ndarray, input_size: int) -> tuple[np.ndarray, LetterboxMeta]:
    _require_hwc_uint8(image, "image")
    if input_size < 32:
        raise ValueError("input_size must be >= 32")
    canvas, meta = letterbox(image, input_size)
    blob = canvas.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None]
    return np.ascontiguousarray(blob), meta


def letterbox(
    image: np.ndarray,
    input_size: int,
    color: int = 114,
) -> tuple[np.ndarray, LetterboxMeta]:
    height, width = image.shape[:2]
    scale = min(input_size / height, input_size / width)
    resized_width = min(int(round(width * scale)), input_size)
    resized_height = min(int(round(height * scale)), input_size)
    resized = resize_image(image, resized_width, resized_height)
    pad_x = (input_size - resized_width) // 2
    pad_y = (input_size - resized_height) // 2
    canvas = np.full((input_size, input_size, image.shape[2]), color, dtype=image.dtype)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    meta = LetterboxMeta(
        scale=float(scale),
        pad_x=pad_x,
        pad_y=pad_y,
        original_width=width,
        original_height=height,
        input_size=input_size,
    )
    return canvas, meta


def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.shape[1] == width and image.shape[0] == height:
        return image
    if cv2 is not None:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return _resize_nearest(image, width, height)


def iter_image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"calibration directory does not exist: {directory}")
    paths = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES | {".npy"}
    ]
    if not paths:
        raise FileNotFoundError(f"no calibration images found in {directory}")
    return sorted(paths)


def _require_hwc_uint8(image: np.ndarray, label: str) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"{label} must be a uint8 HWC RGB array")


def _resize_nearest(image: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    y_index = np.minimum(
        (np.arange(height) * source_height / height).astype(np.int64),
        source_height - 1,
    )
    x_index = np.minimum(
        (np.arange(width) * source_width / width).astype(np.int64),
        source_width - 1,
    )
    return image[y_index][:, x_index]
