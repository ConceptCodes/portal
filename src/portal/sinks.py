import abc
import subprocess
from pathlib import Path

import cv2
import numpy as np


class VideoSink(abc.ABC):
    @abc.abstractmethod
    def write(self, frame: np.ndarray) -> None: ...

    @abc.abstractmethod
    def close(self) -> None: ...


class DisplaySink(VideoSink):
    def __init__(self, title: str = "Portal - Cropped") -> None:
        self._title = title

    def write(self, frame: np.ndarray) -> None:
        cv2.imshow(self._title, frame)
        cv2.waitKey(1)

    def close(self) -> None:
        cv2.destroyAllWindows()


class FileSink(VideoSink):
    def __init__(self, path: str | Path, fps: float, width: int, height: int) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # pyright: ignore[reportAttributeAccessIssue]
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open video writer for: {path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()


class StreamSink(VideoSink):
    def __init__(self, url: str, width: int, height: int, fps: float = 30) -> None:
        output_format = self._detect_format(url)
        self._process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-f",
                output_format,
                url,
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    @staticmethod
    def _detect_format(url: str) -> str:
        if url.startswith("rtmp://"):
            return "flv"
        if url.startswith("udp://"):
            return "mpegts"
        if url.startswith("srt://"):
            return "mpegts"
        if url.startswith("rtsp://"):
            return "rtsp"
        return "mp4"

    def write(self, frame: np.ndarray) -> None:
        if self._process.stdin is None:
            raise RuntimeError("Stream process stdin is closed")
        self._process.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        self._process.wait(timeout=10)


class NDISink(VideoSink):
    def __init__(self, name: str, width: int, height: int, fps: float = 30) -> None:
        self._width = width
        self._height = height
        self._fps = fps

        try:
            import NDIlib  # pyright: ignore[reportMissingImports]

            self._ndi = NDIlib
        except ImportError:
            raise RuntimeError(
                "NDI output requires the NDI SDK and ndi-python. "
                "Install with: pip install ndi-python\n"
                "The NDI SDK is available from: https://ndi.video/tools/ndi-sdk/"
            )

        if not self._ndi.initialize():
            raise RuntimeError("Failed to initialize NDI")

        send_desc = self._ndi.send_create_t()
        send_desc.ndi_name = name
        self._send = self._ndi.send_create(send_desc)
        if self._send is None:
            raise RuntimeError("Failed to create NDI send instance")

        self._frame_data = None
        self._bgra_buf: np.ndarray | None = None

    def write(self, frame: np.ndarray) -> None:
        if self._bgra_buf is None or self._bgra_buf.shape[:2] != frame.shape[:2]:
            self._bgra_buf = np.empty((*frame.shape[:2], 4), dtype=np.uint8)
        cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA, dst=self._bgra_buf)
        video_frame = self._ndi.VideoFrameV2()
        video_frame.data = self._bgra_buf.tobytes()
        video_frame.FourCC = self._ndi.FOURCC_VIDEO_TYPE_BGRA
        video_frame.xres = self._bgra_buf.shape[1]
        video_frame.yres = self._bgra_buf.shape[0]
        video_frame.frame_rate_D = 1
        video_frame.frame_rate_N = int(self._fps)
        video_frame.picture_aspect_ratio = self._width / self._height
        self._ndi.send_send_video_async_v2(self._send, video_frame)

    def close(self) -> None:
        if self._send is not None:
            self._ndi.send_destroy(self._send)
        self._ndi.destroy()


class CompositeSink(VideoSink):
    def __init__(self, sinks: list[VideoSink]) -> None:
        self._sinks = sinks

    def write(self, frame: np.ndarray) -> None:
        for sink in self._sinks:
            sink.write(frame)

    def close(self) -> None:
        for sink in self._sinks:
            sink.close()
