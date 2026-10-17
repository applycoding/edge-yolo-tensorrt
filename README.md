# Edge YOLO TensorRT

Fine-tune an Ultralytics YOLO detector and deploy the same graph through ONNX Runtime or TensorRT. The training config targets coco128. Engine build and the benchmark run on the machine you point them at, including a Jetson.

Arpan Bom

![CI](https://github.com/applycoding/edge-yolo-tensorrt/actions/workflows/ci.yml/badge.svg)

## Pipeline

Train on a workstation. Export ONNX there. Build the TensorRT engine on the Jetson that will serve the camera, then measure that device.

```
coco128.yaml
    │  edge-yolo train
    ▼
best.pt
    │  edge-yolo export-onnx
    ▼
model.onnx ──────► OnnxRuntimeDetector
    │  edge-yolo export-trt   (FP16, or INT8 + entropy calibration)
    ▼
model.engine ────► TensorRtDetector
                      │
                      ▼
                   edge-yolo bench
```

`OnnxRuntimeDetector` and `TensorRtDetector` both implement `Detector`: letterbox, a single `infer` call, then shared decode and NMS. Latency from `bench` includes that whole path.

## Layout

| Path | Role |
| --- | --- |
| `configs/coco128.yaml` | Fine-tune settings for the Ultralytics coco128 subset |
| `configs/export.yaml` | ONNX and engine paths |
| `src/edge_yolo/train.py` | Training entry |
| `src/edge_yolo/export_onnx.py` | Ultralytics ONNX export |
| `src/edge_yolo/export_tensorrt.py` | TensorRT FP16 / INT8 builder |
| `src/edge_yolo/calibrate.py` | INT8 batch source and entropy calibrator |
| `src/edge_yolo/detectors/` | Shared detector interface, ONNX Runtime, TensorRT |
| `src/edge_yolo/benchmark.py` | FPS and latency measurement |
| `Dockerfile` | Jetson image, `nvcr.io/nvidia/l4t-base` |
| `docs/jetson.md` | Flash, clocks, wheels, calibration, run commands |

## Install

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Workstation extras, when you are ready to train or run ONNX:

```bash
pip install -e ".[train,onnx,media]"
```

TensorRT is not a pip extra. On Jetson it comes from `python3-libnvinfer`. Details are in [docs/jetson.md](docs/jetson.md).

## Train

`configs/coco128.yaml` fine-tunes `yolov8n.pt` on `coco128.yaml` for 20 epochs at 640. coco128 is the public 128-image COCO subset shipped with Ultralytics. The first run downloads that subset and the checkpoint. Set `fraction` below 1 to use a smaller slice of the same set.

```bash
python -m edge_yolo train --config configs/coco128.yaml
```

Weights land in `runs/detect/coco128/weights/best.pt`.

## Export

```bash
python -m edge_yolo export-onnx --config configs/export.yaml

python -m edge_yolo export-trt \
  --onnx artifacts/coco128.onnx \
  --precision fp16 \
  --output artifacts/coco128_fp16.engine

python -m edge_yolo export-trt \
  --onnx artifacts/coco128.onnx \
  --precision int8 \
  --output artifacts/coco128_int8.engine \
  --calibration-dir data/calibration \
  --calibration-cache artifacts/coco128_int8.cache
```

INT8 uses TensorRT's entropy calibrator. Put representative camera frames in `data/calibration` (PNG, JPEG, or uint8 HWC `.npy`). The builder also enables FP16 so layers without an INT8 kernel can fall back. The engine is batch 1, static shape. Build it on the target GPU.

## Inference

```bash
python -m edge_yolo infer \
  --backend onnx \
  --model artifacts/coco128.onnx \
  --image frame.npy
```

`--backend tensorrt` takes an `.engine` file. Images are RGB uint8 HWC. `.npy` files load with NumPy; PNG and JPEG need the `media` extra (Pillow). The default decode handles Ultralytics YOLOv8 raw output and end-to-end NMS output. Pass `--layout yolov5` for a YOLOv5 head with objectness.

## Benchmark

`bench` measures the device it is running on. It does not ship or reuse numbers from somewhere else.

```bash
python -m edge_yolo bench \
  --backend tensorrt \
  --model artifacts/coco128_fp16.engine \
  --precision fp16 \
  --input-size 640 \
  --warmup 20 \
  --iterations 200 \
  --json results/device.json
```

`--precision` is the label stored beside the samples. The engine file is what actually ran. Omit `--images` to feed synthetic frames, or pass a directory of real frames. The command prints a markdown row and, with `--json`, writes mean, p50, p95, p99, min, max, and FPS.

Paste that row into the table below. Do not type a number the command did not print.

## Jetson

The container starts from L4T base (`nvcr.io/nvidia/l4t-base`, default tag `r36.2.0` / JetPack 6.0). Match the tag to `nv_tegra_release` on the board. Power mode, PyTorch wheel order, ONNX Runtime on Tegra, and the exact `docker run` lines are in [docs/jetson.md](docs/jetson.md).

```bash
docker build --build-arg L4T_VERSION=r36.2.0 -t edge-yolo:l4t .
```

## Results

This table is empty on purpose. A cell is filled only with a measurement from `edge-yolo bench` on that device.

| Device | Backend | Precision | Input | FPS | p50 ms | p95 ms | Date |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | |

## Tests

Unit tests use synthetic tensors and stand-ins for ONNX Runtime, TensorRT, and Ultralytics. CI does not need a GPU and does not download coco128 or weights.

```bash
ruff check .
ruff format --check .
pytest
```

GitHub Actions runs those three commands on every pull request.

## License

MIT. See [LICENSE](LICENSE).
