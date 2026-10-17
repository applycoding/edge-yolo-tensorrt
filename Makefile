.PHONY: lint test train export-onnx export-trt bench

lint:
	ruff check .
	ruff format --check .

test:
	pytest

train:
	python -m edge_yolo train --config configs/coco128.yaml

export-onnx:
	python -m edge_yolo export-onnx --config configs/export.yaml

export-trt:
	python -m edge_yolo export-trt --config configs/export.yaml --precision fp16

bench:
	python -m edge_yolo bench --backend onnx --model artifacts/coco128.onnx --json results/onnx.json
