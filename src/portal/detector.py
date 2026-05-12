from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]


class PortalError(Exception):
    """Base exception for all portal errors."""


class ModelError(PortalError):
    """Raised when model loading or inference fails."""


class VideoError(PortalError):
    """Raised when video input/output operations fail."""


_TRACKER_CONFIG = "bytetrack.yaml"


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    track_id: int | None = None


class PersonDetector:
    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        self._device = _resolve_device()
        try:
            self._model = YOLO(model_name)
        except Exception as exc:
            raise ModelError(f"Failed to load model '{model_name}': {exc}") from exc
        self._track_ages: dict[int, int] = {}

    @property
    def device(self) -> str:
        return self._device

    @property
    def track_ages(self) -> dict[int, int]:
        return self._track_ages

    @property
    def model_name(self) -> str:
        name = self._model.model_name
        return name if name is not None else ""

    def warmup(self) -> None:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(2):
            self._model(dummy, device=self._device, verbose=False)

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> Sequence[Detection]:
        try:
            results = self._model.track(
                frame,
                classes=[0],
                conf=conf_threshold,
                persist=True,
                tracker=_TRACKER_CONFIG,
                device=self._device,
                verbose=False,
            )
        except Exception as exc:
            raise ModelError(f"Inference failed: {exc}") from exc
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().int().numpy()  # type: ignore[union-attr]
            confs = boxes.conf.cpu().numpy()  # type: ignore[union-attr]
            ids = boxes.id.cpu().int().numpy()  # type: ignore[union-attr]
            for i in range(len(boxes)):
                x1, y1, x2, y2 = xyxy[i].tolist()
                detections.append(
                    Detection(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=float(confs[i]),
                        track_id=int(ids[i]),
                    )
                )

        current_ids: set[int] = set()
        for d in detections:
            if d.track_id is not None:
                current_ids.add(d.track_id)
                self._track_ages[d.track_id] = self._track_ages.get(d.track_id, 0) + 1
        for tid in list(self._track_ages):
            if tid not in current_ids:
                del self._track_ages[tid]

        return detections
