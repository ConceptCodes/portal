import numpy as np
import pytest

from portal.sinks import DisplaySink, FileSink, StreamSink, VideoSink


class TestVideoSink:
    def test_abstract_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            VideoSink()  # type: ignore[abstract]


class TestDisplaySink:
    def test_init(self) -> None:
        sink = DisplaySink()
        assert sink is not None

    def test_write_does_not_crash(self) -> None:
        sink = DisplaySink()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        sink.write(frame)
        sink.close()


class TestFileSink:
    def test_write_creates_file(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "test.mp4"
        sink = FileSink(path, fps=30, width=100, height=100)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        sink.write(frame)
        sink.close()
        assert path.exists()
        assert path.stat().st_size > 0

    def test_bad_codec_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "test.xyz"
        with pytest.raises(RuntimeError):
            FileSink(path, fps=30, width=100, height=100)

    def test_parent_dir_created(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "sub" / "test.mp4"
        FileSink(path, fps=30, width=100, height=100).close()
        assert path.parent.exists()


class TestStreamSink:
    def test_detect_format_rtmp(self) -> None:
        assert StreamSink._detect_format("rtmp://example.com/live") == "flv"

    def test_detect_format_udp(self) -> None:
        assert StreamSink._detect_format("udp://239.0.0.1:1234") == "mpegts"

    def test_detect_format_srt(self) -> None:
        assert StreamSink._detect_format("srt://example.com:1234") == "mpegts"

    def test_detect_format_rtsp(self) -> None:
        assert StreamSink._detect_format("rtsp://example.com/stream") == "rtsp"

    def test_detect_format_default(self) -> None:
        assert StreamSink._detect_format("/tmp/out.mp4") == "mp4"

    def test_init_no_url_does_not_crash(self) -> None:
        sink = StreamSink("/dev/null", width=100, height=100, fps=10)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        sink.write(frame)
        sink.close()


class TestNDISink:
    def test_init_fails_without_sdk(self) -> None:
        with pytest.raises(RuntimeError, match="NDI"):
            from portal.sinks import NDISink

            NDISink("test", 1280, 720)
