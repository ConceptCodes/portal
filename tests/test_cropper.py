import cv2
import numpy as np
import pytest

from portal.cropper import BoxSmoother, crop_frame, draw_detections, select_primary_subject
from portal.detector import Detection


def _make_det(
    x1: int = 0,
    y1: int = 0,
    x2: int = 100,
    y2: int = 100,
    confidence: float = 0.9,
    track_id: int | None = 1,
) -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence, track_id=track_id)


class TestSelectPrimarySubject:
    def test_empty_detections(self) -> None:
        assert select_primary_subject([], 640, 480) is None

    def test_single_detection(self) -> None:
        d = _make_det()
        result = select_primary_subject([d], 640, 480)
        assert result is d

    def test_lock_track_id_found(self) -> None:
        d1 = _make_det(track_id=1)
        d2 = _make_det(track_id=2)
        result = select_primary_subject([d1, d2], 640, 480, lock_track_id=2)
        assert result is d2

    def test_lock_track_id_not_found(self) -> None:
        d1 = _make_det(track_id=1)
        result = select_primary_subject([d1], 640, 480, lock_track_id=99)
        assert result is d1

    def test_larger_area_wins(self) -> None:
        small = _make_det(x1=0, y1=0, x2=50, y2=50, track_id=1)
        large = _make_det(x1=0, y1=0, x2=200, y2=200, track_id=2)
        result = select_primary_subject([small, large], 640, 480)
        assert result is large

    def test_central_detection_preferred(self) -> None:
        edge = _make_det(x1=0, y1=0, x2=50, y2=50, track_id=1)
        center = _make_det(x1=295, y1=215, x2=345, y2=265, track_id=2)
        result = select_primary_subject([edge, center], 640, 480)
        assert result is center

    def test_older_track_preferred(self) -> None:
        young = _make_det(track_id=1)
        old = _make_det(track_id=2)
        result = select_primary_subject(
            [young, old],
            640,
            480,
            track_ages={1: 1, 2: 100},
        )
        assert result is old

    def test_no_track_ages_fallback(self) -> None:
        d1 = _make_det(track_id=1)
        d2 = _make_det(track_id=2)
        result = select_primary_subject([d1, d2], 640, 480, track_ages=None)
        assert result is not None

    def test_detections_without_track_id(self) -> None:
        d = _make_det(track_id=None)
        result = select_primary_subject([d], 640, 480)
        assert result is d


class TestBoxSmoother:
    def test_first_update_returns_detection_box(self) -> None:
        smoother = BoxSmoother(padding=0.0)
        d = _make_det(x1=100, y1=100, x2=300, y2=300)
        box = smoother.update(d, 640, 480)
        x1, y1, x2, y2 = box
        assert x1 == 100
        assert y1 == 100
        assert x2 == 300
        assert y2 == 300

    def test_no_detection_coast_returns_last_box(self) -> None:
        smoother = BoxSmoother(padding=0.0, max_coast=30)
        d = _make_det(x1=100, y1=100, x2=300, y2=300)
        smoother.update(d, 640, 480)
        for _ in range(5):
            box = smoother.update(None, 640, 480)
            assert box != (0, 0, 640, 480)

    def test_coast_exceeded_returns_full_frame(self) -> None:
        smoother = BoxSmoother(padding=0.0, max_coast=5)
        d = _make_det(x1=100, y1=100, x2=300, y2=300)
        smoother.update(d, 640, 480)
        for _ in range(10):
            smoother.update(None, 640, 480)
        box = smoother.update(None, 640, 480)
        assert box == (0, 0, 640, 480)

    def test_coast_before_first_detection_returns_full_frame(self) -> None:
        smoother = BoxSmoother(padding=0.0)
        box = smoother.update(None, 640, 480)
        assert box == (0, 0, 640, 480)

    def test_smoothing_converges(self) -> None:
        smoother = BoxSmoother(padding=0.0, alpha=0.5)
        d = _make_det(x1=0, y1=0, x2=200, y2=200)
        first = smoother.update(d, 640, 480)
        d2 = _make_det(x1=100, y1=100, x2=300, y2=300)
        second = smoother.update(d2, 640, 480)
        assert first != second
        cx1 = (first[0] + first[2]) / 2
        cx2 = (second[0] + second[2]) / 2
        assert cx1 != cx2

    def test_jump_detection_snaps(self) -> None:
        smoother = BoxSmoother(padding=0.0, alpha=0.5, jump_threshold=0.1)
        d1 = _make_det(x1=0, y1=0, x2=50, y2=50)
        smoother.update(d1, 640, 480)
        d2 = _make_det(x1=500, y1=500, x2=600, y2=600)
        box = smoother.update(d2, 640, 480)
        cx = (box[0] + box[2]) / 2
        expected_cx = (500 + 600) / 2
        assert abs(cx - expected_cx) < 1

    def test_reset_clears_state(self) -> None:
        smoother = BoxSmoother(padding=0.0)
        smoother.update(_make_det(), 640, 480)
        smoother.reset()
        box = smoother.update(None, 640, 480)
        assert box == (0, 0, 640, 480)

    def test_padding_expands_box(self) -> None:
        smoother = BoxSmoother(padding=0.5)
        d = _make_det(x1=200, y1=200, x2=400, y2=400)
        box = smoother.update(d, 640, 480)
        x1, y1, x2, y2 = box
        assert x1 < 200
        assert y1 < 200
        assert x2 > 400
        assert y2 > 400

    def test_box_clamped_to_frame(self) -> None:
        smoother = BoxSmoother(padding=1.0)
        d = _make_det(x1=0, y1=0, x2=50, y2=50)
        box = smoother.update(d, 640, 480)
        x1, y1, x2, y2 = box
        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= 640
        assert y2 <= 480


class TestCropFrame:
    def test_no_crop_box_returns_full_frame(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = crop_frame(frame, output_width=640, output_height=480)
        assert result.shape[:2] == (480, 640)

    def test_crop_box(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[100:200, 100:200] = 255
        result = crop_frame(frame, (100, 100, 200, 200), output_width=200, output_height=200)
        assert result.shape[:2] == (200, 200)
        r = cv2.resize(frame[100:200, 100:200], (200, 200))
        assert np.array_equal(result, r)

    def test_resize_to_output(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = crop_frame(frame, (0, 0, 320, 240), output_width=1280, output_height=720)
        assert result.shape[:2] == (720, 1280)

    def test_same_size_no_copy(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = crop_frame(frame, output_width=1280, output_height=720)
        assert result.shape[:2] == (720, 1280)

    def test_preallocated_buffer_reused(self) -> None:
        from portal.cropper import _crop_buffers

        if hasattr(_crop_buffers, "buf"):
            del _crop_buffers.buf
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result1 = crop_frame(frame, (0, 0, 320, 240), output_width=1280, output_height=720)
        assert hasattr(_crop_buffers, "buf")
        assert _crop_buffers.buf is result1
        result2 = crop_frame(frame, (0, 0, 320, 240), output_width=1280, output_height=720)
        assert result2 is result1

    def test_different_size_allocates_new_buffer(self) -> None:
        from portal.cropper import _crop_buffers

        if hasattr(_crop_buffers, "buf"):
            del _crop_buffers.buf
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        r1 = crop_frame(frame, (0, 0, 320, 240), output_width=640, output_height=360)
        r2 = crop_frame(frame, (0, 0, 320, 240), output_width=1280, output_height=720)
        assert r2 is not r1


class TestDrawDetections:
    def test_returns_copy(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_detections(frame, [])
        assert result is not frame

    def test_empty_detections(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_detections(frame, [])
        assert np.array_equal(result, frame)

    def test_draws_rectangle(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        d = _make_det(x1=10, y1=10, x2=50, y2=50, track_id=1)
        result = draw_detections(frame, [d])
        assert not np.array_equal(result, frame)

    def test_multiple_detections(self) -> None:
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        d1 = _make_det(x1=10, y1=10, x2=50, y2=50, track_id=1)
        d2 = _make_det(x1=100, y1=100, x2=150, y2=150, track_id=2)
        result = draw_detections(frame, [d1, d2])
        assert not np.array_equal(result, frame)
