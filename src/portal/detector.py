from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO


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
        self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._model = YOLO(model_name)
        self._track_ages: dict[int, int] = {}

    @property
    def device(self) -> str:
        return self._device

    @property
    def track_ages(self) -> dict[int, int]:
        return self._track_ages

    @property
    def model_name(self) -> str:
        return self._model.model_name

    def warmup(self) -> None:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model(dummy, device=self._device, verbose=False)
        self._model(dummy, device=self._device, verbose=False)

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> Sequence[Detection]:
        results = self._model.track(
            frame,
            classes=[0],
            conf=conf_threshold,
            persist=True,
            tracker="bytetrack.yaml",
            device=self._device,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            if result.boxes is not None and result.boxes.id is not None:
                for box, track_id in zip(result.boxes, result.boxes.id):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    tid = int(track_id.item())
                    detections.append(
                        Detection(
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            confidence=float(box.conf[0]),
                            track_id=tid,
                        )
                    )

        current_ids = {d.track_id for d in detections if d.track_id is not None}
        for tid in list(self._track_ages.keys()):
            if tid in current_ids:
                self._track_ages[tid] += 1
            else:
                del self._track_ages[tid]
        for tid in current_ids:
            if tid not in self._track_ages:
                self._track_ages[tid] = 1

        return detections
