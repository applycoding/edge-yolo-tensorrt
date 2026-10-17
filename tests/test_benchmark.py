import json
from pathlib import Path

import numpy as np
import pytest

from edge_yolo.benchmark import (
    format_markdown_row,
    run_benchmark,
    summarize_latencies,
    synthetic_images,
    write_report,
)


def test_latency_summary_uses_linear_percentiles():
    stats = summarize_latencies([10, 20, 30, 40, 50])
    assert stats.iterations == 5
    assert stats.mean_ms == pytest.approx(30)
    assert stats.p50_ms == pytest.approx(30)
    assert stats.p95_ms == pytest.approx(48)
    assert stats.p99_ms == pytest.approx(49.6)
    assert stats.min_ms == pytest.approx(10)
    assert stats.max_ms == pytest.approx(50)
    assert stats.fps == pytest.approx(1000 / 30)


def test_benchmark_discards_warmup_and_records_the_device():
    calls = {"count": 0}

    def predict(image: np.ndarray) -> list[object]:
        assert image.dtype == np.uint8
        calls["count"] += 1
        return []

    images = synthetic_images(2, 8, 8, seed=1)
    report = run_benchmark(
        predict,
        images,
        warmup=2,
        iterations=3,
        backend="onnx",
        precision="fp16",
        device="unit-test",
        input_size=32,
        measured_on="2026-10-17",
    )
    assert calls["count"] == 5
    assert report.latency.iterations == 3
    assert report.device == "unit-test"
    assert report.measured_on == "2026-10-17"
    row = format_markdown_row(report)
    assert row.split("|")[1].strip() == "unit-test"
    assert "2026-10-17" in row


def test_write_report_roundtrip(tmp_path: Path):
    images = [np.zeros((4, 4, 3), dtype=np.uint8)]
    report = run_benchmark(
        lambda _image: [],
        images,
        warmup=0,
        iterations=1,
        backend="tensorrt",
        precision="int8",
        device="cpu",
        input_size=64,
        measured_on="2026-10-17",
    )
    path = tmp_path / "results" / "report.json"
    write_report(report, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["backend"] == "tensorrt"
    assert payload["precision"] == "int8"
    assert payload["fps"] == pytest.approx(report.latency.fps)
    assert payload["iterations"] == 1


def test_benchmark_rejects_empty_work():
    with pytest.raises(ValueError):
        summarize_latencies([])
    with pytest.raises(ValueError):
        run_benchmark(
            lambda _image: [],
            [],
            warmup=0,
            iterations=1,
            backend="onnx",
            precision="fp32",
            device="cpu",
            input_size=32,
        )
