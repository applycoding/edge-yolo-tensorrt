"""Optional runtimes.

Training, ONNX Runtime, and TensorRT are imported defensively so the package
can be imported in CI without a GPU, CUDA, or those wheels.
"""

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import tensorrt as trt
except ImportError:
    trt = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None


def require_ultralytics():
    if YOLO is None:
        raise RuntimeError(
            "ultralytics is not installed. Install the train extra: pip install 'edge-yolo[train]'"
        )
    return YOLO


def require_tensorrt():
    if trt is None:
        raise RuntimeError(
            "tensorrt is not installed. On Jetson, install python3-libnvinfer "
            "from the L4T apt repository."
        )
    return trt


def require_onnxruntime():
    if ort is None:
        raise RuntimeError(
            "onnxruntime is not installed. Install the onnx extra: pip install 'edge-yolo[onnx]'"
        )
    return ort
