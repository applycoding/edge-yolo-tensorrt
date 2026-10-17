import argparse
import json
import sys
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import assert_never

from edge_yolo.benchmark import format_markdown_row, run_benchmark, synthetic_images, write_report
from edge_yolo.config import load_export_config, load_train_config
from edge_yolo.detectors import open_detector
from edge_yolo.device import probe_device
from edge_yolo.export_onnx import export_onnx
from edge_yolo.export_tensorrt import build_engine_from_images
from edge_yolo.preprocess import iter_image_paths, load_rgb_image
from edge_yolo.train import train_from_config
from edge_yolo.types import Backend, OutputLayout, Precision


class Command(Enum):
    TRAIN = "train"
    EXPORT_ONNX = "export-onnx"
    EXPORT_TRT = "export-trt"
    INFER = "infer"
    BENCH = "bench"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edge-yolo",
        description="Fine-tune YOLO and deploy it with ONNX Runtime or TensorRT.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="fine-tune a detector from a YAML config")
    train.add_argument("--config", type=Path, required=True)

    export_onnx_parser = subparsers.add_parser("export-onnx", help="export PyTorch weights to ONNX")
    export_onnx_parser.add_argument("--config", type=Path)
    export_onnx_parser.add_argument("--weights", type=Path)
    export_onnx_parser.add_argument("--output", type=Path)
    export_onnx_parser.add_argument("--imgsz", type=int)
    export_onnx_parser.add_argument("--opset", type=int)
    export_onnx_parser.add_argument("--dynamic", action="store_true")
    export_onnx_parser.add_argument("--no-simplify", action="store_true")

    export_trt_parser = subparsers.add_parser(
        "export-trt",
        help="build an FP16 or INT8 TensorRT engine",
    )
    export_trt_parser.add_argument("--config", type=Path)
    export_trt_parser.add_argument("--onnx", type=Path)
    export_trt_parser.add_argument("--output", type=Path)
    export_trt_parser.add_argument(
        "--precision",
        choices=[item.value for item in Precision],
        default=Precision.FP16.value,
    )
    export_trt_parser.add_argument("--imgsz", type=int)
    export_trt_parser.add_argument("--workspace-mib", type=int)
    export_trt_parser.add_argument("--calibration-dir", type=Path)
    export_trt_parser.add_argument("--calibration-cache", type=Path)
    export_trt_parser.add_argument("--batch-size", type=int, default=1)

    infer = subparsers.add_parser("infer", help="run one image through ONNX Runtime or TensorRT")
    _add_runtime_args(infer)
    infer.add_argument("--image", type=Path, required=True)

    bench = subparsers.add_parser("bench", help="measure end-to-end FPS and latency on this device")
    _add_runtime_args(bench)
    bench.add_argument(
        "--precision",
        choices=[item.value for item in Precision],
        default=Precision.FP32.value,
        help="label recorded with the measurement; it does not convert the model",
    )
    bench.add_argument("--warmup", type=int, default=10)
    bench.add_argument("--iterations", type=int, default=100)
    bench.add_argument(
        "--images",
        type=Path,
        help="directory of images; synthetic frames are used if omitted",
    )
    bench.add_argument("--images-count", type=int, default=8)
    bench.add_argument("--height", type=int, default=640)
    bench.add_argument("--width", type=int, default=640)
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--json", type=Path)
    bench.add_argument("--device", help="override the probed device label stored in the report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        command = Command(args.command)
        match command:
            case Command.TRAIN:
                return _train(args)
            case Command.EXPORT_ONNX:
                return _export_onnx(args)
            case Command.EXPORT_TRT:
                return _export_trt(args)
            case Command.INFER:
                return _infer(args)
            case Command.BENCH:
                return _bench(args)
            case _:
                assert_never(command)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def cli() -> None:
    sys.exit(main())


def _train(args: argparse.Namespace) -> int:
    config = load_train_config(args.config)
    summary = train_from_config(config)
    print(f"save_dir: {summary.save_dir}")
    print(f"best: {summary.best_weights}")
    return 0


def _export_onnx(args: argparse.Namespace) -> int:
    config = load_export_config(args.config) if args.config else None
    weights = args.weights or (Path(config.weights) if config else None)
    output = args.output or (Path(config.onnx_path) if config else None)
    if weights is None or output is None:
        raise ValueError("export-onnx needs --weights and --output, or --config")
    imgsz = _pick(args.imgsz, config.imgsz if config else None, 640)
    opset = _pick(args.opset, config.opset if config else None, 17)
    simplify = False if args.no_simplify else (config.simplify if config else True)
    dynamic = bool(args.dynamic or (config.dynamic if config else False))
    path = export_onnx(
        weights,
        output,
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=dynamic,
    )
    print(path)
    return 0


def _export_trt(args: argparse.Namespace) -> int:
    config = load_export_config(args.config) if args.config else None
    precision = Precision(args.precision)
    onnx_path = args.onnx or (Path(config.onnx_path) if config else None)
    if onnx_path is None:
        raise ValueError("export-trt needs --onnx or --config")
    if args.output is not None:
        output = args.output
    elif config is not None:
        output = Path(config.engine_path(precision.value))
    else:
        raise ValueError("export-trt needs --output or --config")
    imgsz = _pick(args.imgsz, config.imgsz if config else None, 640)
    workspace_mib = _pick(args.workspace_mib, config.workspace_mib if config else None, 2048)
    calibration_dir = args.calibration_dir
    if calibration_dir is None and config is not None and precision is Precision.INT8:
        calibration_dir = Path(config.calibration_dir)
    if args.calibration_cache is not None:
        calibration_cache = args.calibration_cache
    elif config is not None:
        calibration_cache = Path(config.calibration_cache)
    else:
        calibration_cache = Path("artifacts/int8.cache")
    path = build_engine_from_images(
        onnx_path=onnx_path,
        output_path=output,
        precision=precision,
        calibration_dir=calibration_dir,
        calibration_cache=calibration_cache,
        imgsz=imgsz,
        workspace_mib=workspace_mib,
        batch_size=args.batch_size,
    )
    print(path)
    return 0


def _infer(args: argparse.Namespace) -> int:
    detector = _detector_from_args(args)
    detections = detector.predict(load_rgb_image(args.image))
    print(json.dumps([item.to_dict() for item in detections], indent=2))
    return 0


def _bench(args: argparse.Namespace) -> int:
    detector = _detector_from_args(args)
    if args.images is not None:
        images = [load_rgb_image(path) for path in iter_image_paths(args.images)]
    else:
        images = synthetic_images(args.images_count, args.height, args.width, seed=args.seed)
    device = args.device or probe_device().label
    report = run_benchmark(
        detector.predict,
        images,
        warmup=args.warmup,
        iterations=args.iterations,
        backend=detector.backend_name,
        precision=Precision(args.precision).value,
        device=device,
        input_size=args.input_size,
    )
    print(format_markdown_row(report))
    print(json.dumps(report.to_dict(), indent=2))
    if args.json is not None:
        write_report(report, args.json)
    return 0


def _detector_from_args(args: argparse.Namespace):
    return open_detector(
        Backend(args.backend),
        args.model,
        input_size=args.input_size,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        layout=OutputLayout(args.layout),
        providers=args.providers,
    )


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", required=True, choices=[item.value for item in Backend])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--layout",
        default=OutputLayout.AUTO.value,
        choices=[item.value for item in OutputLayout],
    )
    parser.add_argument("--provider", action="append", dest="providers")


def _pick(override: int | None, configured: int | None, default: int) -> int:
    if override is not None:
        return override
    if configured is not None:
        return configured
    return default
