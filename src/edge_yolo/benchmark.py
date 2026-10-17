import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class LatencyStats:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    fps: float
    iterations: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.mean_ms,
            "p50": self.p50_ms,
            "p95": self.p95_ms,
            "p99": self.p99_ms,
            "min": self.min_ms,
            "max": self.max_ms,
            "fps": self.fps,
            "iterations": self.iterations,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    backend: str
    precision: str
    device: str
    input_size: int
    warmup: int
    latency: LatencyStats
    measured_on: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "precision": self.precision,
            "device": self.device,
            "input_size": self.input_size,
            "warmup": self.warmup,
            "iterations": self.latency.iterations,
            "latency_ms": self.latency.to_dict(),
            "fps": self.latency.fps,
            "measured_on": self.measured_on,
        }


def summarize_latencies(samples_ms: Sequence[float]) -> LatencyStats:
    if not samples_ms:
        raise ValueError("latency summary needs at least one sample")
    samples = np.asarray(list(samples_ms), dtype=np.float64)
    mean_ms = float(samples.mean())
    fps = 1000.0 / mean_ms if mean_ms > 0 else float("inf")
    return LatencyStats(
        mean_ms=mean_ms,
        p50_ms=_percentile(samples, 50),
        p95_ms=_percentile(samples, 95),
        p99_ms=_percentile(samples, 99),
        min_ms=float(samples.min()),
        max_ms=float(samples.max()),
        fps=fps,
        iterations=int(samples.size),
    )


def _percentile(samples: np.ndarray, q: float) -> float:
    return float(np.quantile(samples, q / 100.0, method="linear"))


def run_benchmark(
    predict: Callable[[np.ndarray], object],
    images: Sequence[np.ndarray],
    *,
    warmup: int,
    iterations: int,
    backend: str,
    precision: str,
    device: str,
    input_size: int,
    measured_on: str | None = None,
) -> BenchmarkReport:
    """Time end-to-end predict calls. Warmup iterations are executed and discarded."""
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if not images:
        raise ValueError("benchmark needs at least one image")

    for index in range(warmup):
        predict(images[index % len(images)])

    samples_ms: list[float] = []
    for index in range(iterations):
        image = images[index % len(images)]
        start = time.perf_counter()
        predict(image)
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    stamp = measured_on or datetime.now(timezone.utc).date().isoformat()
    return BenchmarkReport(
        backend=backend,
        precision=precision,
        device=device,
        input_size=input_size,
        warmup=warmup,
        latency=summarize_latencies(samples_ms),
        measured_on=stamp,
    )


def format_markdown_row(report: BenchmarkReport) -> str:
    latency = report.latency
    return (
        f"| {report.device} | {report.backend} | {report.precision} | {report.input_size} | "
        f"{latency.fps:.2f} | {latency.p50_ms:.2f} | {latency.p95_ms:.2f} | {report.measured_on} |"
    )


def write_report(report: BenchmarkReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")


def synthetic_images(count: int, height: int, width: int, seed: int = 0) -> list[np.ndarray]:
    if count < 1:
        raise ValueError("count must be >= 1")
    if height < 1 or width < 1:
        raise ValueError("height and width must be >= 1")
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8) for _ in range(count)]
