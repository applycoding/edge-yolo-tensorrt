from dataclasses import dataclass
from pathlib import Path

from edge_yolo.config import TrainConfig
from edge_yolo.deps import require_ultralytics


@dataclass(frozen=True, slots=True)
class TrainSummary:
    save_dir: Path
    best_weights: Path


def train_from_config(config: TrainConfig) -> TrainSummary:
    yolo_cls = require_ultralytics()
    model = yolo_cls(config.model)
    results = model.train(**config.to_train_kwargs())
    save_dir = Path(results.save_dir)
    return TrainSummary(save_dir=save_dir, best_weights=save_dir / "weights" / "best.pt")
