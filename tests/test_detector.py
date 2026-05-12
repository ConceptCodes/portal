import numpy as np
import pytest

from portal.detector import Detection, ModelError, PersonDetector, _resolve_device


class TestDetection:
    def test_fields(self) -> None:
        d = Detection(x1=10, y1=20, x2=100, y2=200, confidence=0.85, track_id=3)
        assert d.x1 == 10
        assert d.y1 == 20
        assert d.x2 == 100
        assert d.y2 == 200
        assert d.confidence == 0.85
        assert d.track_id == 3

    def test_default_track_id(self) -> None:
        d = Detection(x1=0, y1=0, x2=10, y2=10, confidence=0.5)
        assert d.track_id is None

    def test_immutable(self) -> None:
        d = Detection(x1=0, y1=0, x2=10, y2=10, confidence=0.5)
        with pytest.raises(AttributeError):
            d.x1 = 99  # type: ignore[misc]


class TestResolveDevice:
    def test_returns_string(self) -> None:
        device = _resolve_device()
        assert device in ("cpu", "mps", "cuda:0")


class TestPersonDetectorInit:
    def test_init_fails_on_bad_model(self) -> None:
        with pytest.raises(ModelError):
            PersonDetector("nonexistent_model_xyz.pt")

    def test_init_with_valid_model(self) -> None:
        detector = PersonDetector("yolov8n.pt")
        assert isinstance(detector.model_name, str)
        assert len(detector.model_name) > 0
        assert detector.device in ("cpu", "mps", "cuda:0")
        assert detector.track_ages == {}

    def test_warmup_completes(self) -> None:
        detector = PersonDetector("yolov8n.pt")
        detector.warmup()

    def test_detect_basic(self) -> None:
        detector = PersonDetector("yolov8n.pt")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(frame, conf_threshold=0.99)
        assert isinstance(results, list)

    def test_track_ages_updates(self) -> None:
        detector = PersonDetector("yolov8n.pt")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.detect(frame, conf_threshold=0.99)
        assert isinstance(detector.track_ages, dict)
