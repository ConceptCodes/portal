import numpy as np
import pytest

from portal.config import ProcessorConfig
from portal.cropper import crop_frame
from portal.detector import Detection, PersonDetector, VideoError
from portal.processor import FileProcessor, LiveProcessor


class TestFileProcessor:
    def test_init(self) -> None:
        detector = PersonDetector("yolov8n.pt")
        config = ProcessorConfig()
        processor = FileProcessor(detector, config)
        assert processor is not None

    def test_process_file_missing_input(self) -> None:
        detector = PersonDetector("yolov8n.pt")
        config = ProcessorConfig()
        processor = FileProcessor(detector, config)
        with pytest.raises(VideoError):
            for _ in processor.process_file("nonexistent.mp4", "out.mp4"):
                pass


class TestLiveProcessor:
    def test_init(self) -> None:
        detector = PersonDetector("yolov8n.pt")
        config = ProcessorConfig()
        processor = LiveProcessor(detector, config)
        assert processor is not None

    def test_bad_camera_raises(self) -> None:
        detector = PersonDetector("yolov8n.pt")
        config = ProcessorConfig()
        processor = LiveProcessor(detector, config)
        with pytest.raises(VideoError):
            processor.run(camera_id=-999)


class TestCropFrameIntegration:
    def test_crop_with_detections(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            Detection(x1=100, y1=100, x2=300, y2=300, confidence=0.9, track_id=1),
        ]
        from portal.cropper import BoxSmoother, select_primary_subject

        primary = select_primary_subject(detections, 640, 480)
        smoother = BoxSmoother(padding=0.1)
        crop_box = smoother.update(primary, 640, 480)
        result = crop_frame(frame, crop_box, output_width=1280, output_height=720)
        assert result.shape[:2] == (720, 1280)
        assert result.dtype == np.uint8

    def test_no_detections_returns_full_frame(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        from portal.cropper import BoxSmoother, select_primary_subject

        primary = select_primary_subject([], 640, 480)
        smoother = BoxSmoother()
        crop_box = smoother.update(primary, 640, 480)
        result = crop_frame(frame, crop_box, output_width=1280, output_height=720)
        assert result.shape[:2] == (720, 1280)
