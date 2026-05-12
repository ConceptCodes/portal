import math
import threading
from collections.abc import Sequence
from functools import lru_cache

import cv2
import numpy as np

from portal.detector import Detection


@lru_cache(maxsize=4)
def _frame_constants(
    frame_width: int, frame_height: int
) -> tuple[float, float, float, float]:
    frame_area = frame_width * frame_height
    fc_x = frame_width / 2.0
    fc_y = frame_height / 2.0
    max_dist = math.hypot(frame_width, frame_height) / 2.0
    return frame_area, fc_x, fc_y, max_dist


def select_primary_subject(
    detections: Sequence[Detection],
    frame_width: int,
    frame_height: int,
    track_ages: dict[int, int] | None = None,
    lock_track_id: int | None = None,
) -> Detection | None:
    if not detections:
        return None

    if lock_track_id is not None:
        for d in detections:
            if d.track_id == lock_track_id:
                return d

    if track_ages is None:
        track_ages = {}

    max_age = max(track_ages.values()) if track_ages else 1
    frame_area, fc_x, fc_y, max_dist = _frame_constants(frame_width, frame_height)

    best_score = -1.0
    best_detection = detections[0]

    for d in detections:
        bw = d.x2 - d.x1
        bh = d.y2 - d.y1
        area_ratio = (bw * bh) / frame_area

        cx = (d.x1 + d.x2) / 2.0
        cy = (d.y1 + d.y2) / 2.0
        center_dist = math.hypot(cx - fc_x, cy - fc_y)
        center_score = 1.0 - center_dist / max_dist

        age = track_ages.get(d.track_id, 1) if d.track_id is not None else 1
        age_ratio = age / max_age

        score = 0.50 * area_ratio + 0.35 * center_score + 0.15 * age_ratio

        if score > best_score:
            best_score = score
            best_detection = d

    return best_detection


class BoxSmoother:
    def __init__(
        self,
        padding: float = 0.10,
        alpha: float = 0.10,
        jump_threshold: float = 0.15,
        max_coast: int = 30,
    ) -> None:
        self._padding = padding
        self._alpha = alpha
        self._jump_threshold = jump_threshold
        self._max_coast = max_coast

        self._smooth_cx: float | None = None
        self._smooth_cy: float | None = None
        self._smooth_w: float | None = None
        self._smooth_h: float | None = None
        self._last_raw_area: float | None = None
        self._coast_count: int = 0

    def reset(self) -> None:
        self._smooth_cx = None
        self._smooth_cy = None
        self._smooth_w = None
        self._smooth_h = None
        self._last_raw_area = None
        self._coast_count = 0

    def update(
        self,
        detection: Detection | None,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int]:
        frame_diag = math.hypot(frame_width, frame_height)

        if detection is None:
            self._coast_count += 1
            if self._coast_count > self._max_coast or self._smooth_cx is None:
                return (0, 0, frame_width, frame_height)
            return self._compute_box(frame_width, frame_height)

        self._coast_count = 0

        bw = detection.x2 - detection.x1
        bh = detection.y2 - detection.y1
        pad_x = int(bw * self._padding)
        pad_y = int(bh * self._padding)
        x1 = max(0, detection.x1 - pad_x)
        y1 = max(0, detection.y1 - pad_y)
        x2 = min(frame_width, detection.x2 + pad_x)
        y2 = min(frame_height, detection.y2 + pad_y)

        raw_cx = (x1 + x2) / 2.0
        raw_cy = (y1 + y2) / 2.0
        raw_w = x2 - x1
        raw_h = y2 - y1
        raw_area = raw_w * raw_h

        if self._smooth_cx is None:
            self._smooth_cx = raw_cx
            self._smooth_cy = raw_cy
            self._smooth_w = raw_w
            self._smooth_h = raw_h
            self._last_raw_area = raw_area
        else:
            assert self._smooth_cx is not None and self._smooth_cy is not None
            scx = self._smooth_cx
            scy = self._smooth_cy
            center_dist = math.hypot(raw_cx - scx, raw_cy - scy)
            alpha = self._alpha
            if center_dist / frame_diag > self._jump_threshold:
                alpha = 1.0

            self._smooth_cx = alpha * raw_cx + (1 - alpha) * scx
            self._smooth_cy = alpha * raw_cy + (1 - alpha) * scy

            if self._last_raw_area is not None:
                lra = self._last_raw_area
                area_change = abs(raw_area - lra) / max(lra, 1)
                if area_change > 0.20:
                    self._smooth_w = raw_w
                    self._smooth_h = raw_h
                    self._last_raw_area = raw_area

        return self._compute_box(frame_width, frame_height)

    def _compute_box(self, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        cx = self._smooth_cx
        cy = self._smooth_cy
        w = self._smooth_w
        h = self._smooth_h
        assert cx is not None and cy is not None and w is not None and h is not None
        x1 = max(0, int(cx - w / 2))
        y1 = max(0, int(cy - h / 2))
        x2 = min(frame_width, int(cx + w / 2))
        y2 = min(frame_height, int(cy + h / 2))
        return (x1, y1, x2, y2)


_crop_buffers: threading.local = threading.local()


def crop_frame(
    frame: np.ndarray,
    crop_box: tuple[int, int, int, int] | None = None,
    output_width: int = 1280,
    output_height: int = 720,
) -> np.ndarray:
    if crop_box is None:
        x1, y1, x2, y2 = 0, 0, frame.shape[1], frame.shape[0]
    else:
        x1, y1, x2, y2 = crop_box

    cropped = frame[y1:y2, x1:x2]

    if cropped.shape[:2] == (output_height, output_width):
        return cropped

    buf: np.ndarray | None = getattr(_crop_buffers, "buf", None)
    if buf is None or buf.shape[:2] != (output_height, output_width):
        buf = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        _crop_buffers.buf = buf

    cv2.resize(cropped, (output_width, output_height), dst=buf)
    return buf


def draw_detections(
    frame: np.ndarray,
    detections: Sequence[Detection],
) -> np.ndarray:
    out = frame.copy()
    for d in detections:
        label = f"ID:{d.track_id}" if d.track_id is not None else f"person {d.confidence:.2f}"
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (d.x1, d.y1 - lh - 4), (d.x1 + lw + 4, d.y1), (0, 255, 0), -1)
        cv2.putText(out, label, (d.x1 + 2, d.y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return out
