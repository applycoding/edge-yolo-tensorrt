# Jetson setup

Build the TensorRT engine on the Jetson that will run it. An engine serialized for one GPU will not load on another, including a workstation GPU used for training.

This path targets JetPack 6 / L4T R36 (Orin Nano, Orin NX, AGX Orin) and the TensorRT 10 Python API. The container base is `nvcr.io/nvidia/l4t-base`.

## Host

1. Flash the board with JetPack 6 from NVIDIA SDK Manager so CUDA, cuDNN, and TensorRT match the L4T release.
2. Confirm the release before building the image:

   ```bash
   cat /etc/nv_tegra_release
   ```

3. Install the NVIDIA container runtime (`nvidia-container-toolkit`) so `--runtime nvidia` can see the GPU.
4. For a measurement, lock a known power mode and record it next to the result:

   ```bash
   sudo nvpmodel -q
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```

   Put the `nvpmodel` mode in the notes when you copy a benchmark row into the README. Max clocks are for measurement, not for a deployed robot.

## Image

From the repo root, on the Jetson:

```bash
docker build --build-arg L4T_VERSION=r36.2.0 -t edge-yolo:l4t .
```

Change `L4T_VERSION` if NGC publishes an `l4t-base` tag that matches a newer JetPack on the board. The tag and the host L4T release need to match.

`python3-libnvinfer` comes from the L4T apt repository baked into the base image. That provides the TensorRT Python bindings. The image does not pip-install PyTorch or Ultralytics, because those wheels would replace the Jetson build of PyTorch.

## PyTorch and Ultralytics

Training and the `.pt` to ONNX step need Ultralytics. ONNX export is usually done on the workstation that trained the model. Do it on the Jetson only when you want the whole pipeline on one board.

Install the PyTorch wheel NVIDIA documents for that JetPack before Ultralytics:

<https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html>

Then install Ultralytics without letting pip replace that wheel:

```bash
python3 -m pip install ultralytics --no-deps
python3 -m pip install "edge-yolo[media]"
```

Install the other Ultralytics dependencies it reports missing, and skip any step that reinstalls `torch` or `torchvision`.

## ONNX Runtime

The PyPI `onnxruntime-gpu` packages are not built for Jetson's Tegra CUDA. Two options:

- Use the TensorRT engine as the deployed runtime (`--backend tensorrt`).
- Install an ONNX Runtime build you compiled for that JetPack, then `python3 -m pip install -e ".[onnx]"` only if the wheel actually matches the board.

The ONNX Runtime detector and the TensorRT detector share `Detector.predict`.

## INT8 calibration

Use images that look like the camera stream, not a blank frame. A hundred images is a reasonable start. PNG, JPEG, and uint8 HWC `.npy` files are accepted (image files need Pillow).

```bash
python3 -m edge_yolo export-trt \
  --onnx artifacts/coco128.onnx \
  --precision int8 \
  --output artifacts/coco128_int8.engine \
  --calibration-dir data/calibration \
  --calibration-cache artifacts/coco128_int8.cache \
  --imgsz 640
```

The calibrator implements TensorRT's entropy calibrator (`IInt8EntropyCalibrator2`). FP16 is also enabled so layers without an INT8 kernel can fall back. The engine input is static NCHW, batch 1, which matches a single camera stream. Delete the cache file to force TensorRT to calibrate again.

A zero-filled `.npy` is enough to check that the command runs. It is not a calibration set.

## Run

```bash
docker run --rm --runtime nvidia --network host \
  -v "$PWD":/work -w /work \
  edge-yolo:l4t export-trt \
  --onnx artifacts/coco128.onnx \
  --precision fp16 \
  --output artifacts/coco128_fp16.engine

docker run --rm --runtime nvidia \
  -v "$PWD":/work -w /work \
  edge-yolo:l4t bench \
  --backend tensorrt \
  --model artifacts/coco128_fp16.engine \
  --precision fp16 \
  --iterations 200 \
  --json results/orin_fp16.json
```

`bench` prints one markdown row and writes a JSON report. Copy that row into the README results table yourself. The benchmark times preprocess, network execution, and postprocess on the calling thread.

## Outside Docker

On a Jetson host Python 3.10:

```bash
sudo apt-get update
sudo apt-get install -y python3-libnvinfer python3-pip libopenblas-dev
python3 -m pip install -e .
python3 -m edge_yolo bench --help
```

Add the Jetson PyTorch wheel and `ultralytics` only when you need training or ONNX export on the device.
