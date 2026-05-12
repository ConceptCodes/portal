from portal.config import ProcessorConfig


class TestProcessorConfig:
    def test_default_values(self) -> None:
        config = ProcessorConfig()
        assert config.model == "yolov8n.pt"
        assert config.conf == 0.5
        assert config.padding == 0.10
        assert config.width == 1280
        assert config.height == 720
        assert config.alpha == 0.10
        assert config.jump_threshold == 0.15
        assert config.track_id is None
        assert config.skip == 0
        assert config.show is False
        assert config.output is None

    def test_custom_values(self) -> None:
        config = ProcessorConfig(
            model="yolo11n.pt",
            conf=0.7,
            padding=0.2,
            width=1920,
            height=1080,
            alpha=0.05,
            jump_threshold=0.2,
            track_id=5,
            skip=2,
            show=True,
            output="out.mp4",
        )
        assert config.model == "yolo11n.pt"
        assert config.conf == 0.7
        assert config.padding == 0.2
        assert config.width == 1920
        assert config.height == 1080
        assert config.alpha == 0.05
        assert config.jump_threshold == 0.2
        assert config.track_id == 5
        assert config.skip == 2
        assert config.show is True
        assert config.output == "out.mp4"
