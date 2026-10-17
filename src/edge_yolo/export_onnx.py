from pathlib import Path

from edge_yolo.deps import require_ultralytics


def export_onnx(
    weights: Path,
    output: Path,
    imgsz: int = 640,
    opset: int = 17,
    simplify: bool = True,
    dynamic: bool = False,
) -> Path:
    """Export Ultralytics weights to ONNX.

    The graph stays FP32. Numeric precision is selected when the TensorRT engine is built.
    """
    if imgsz < 32:
        raise ValueError("imgsz must be >= 32")
    if opset < 12:
        raise ValueError("opset must be >= 12")
    yolo_cls = require_ultralytics()
    model = yolo_cls(str(weights))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            simplify=simplify,
            dynamic=dynamic,
            half=False,
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != output.resolve():
        exported.replace(output)
    return output
