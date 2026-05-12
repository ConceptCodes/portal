import queue
import threading
from collections.abc import Generator, Sequence
from pathlib import Path

import cv2
import numpy as np

from portal.config import ProcessorConfig
from portal.cropper import BoxSmoother, crop_frame, draw_detections, select_primary_subject
from portal.detector import Detection, PersonDetector, VideoError


class FileProcessor:
    def __init__(self, detector: PersonDetector, config: ProcessorConfig) -> None:
        self._detector = detector
        self._config = config

    def process_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> Generator[tuple[int, int], None, None]:
        input_path = Path(input_path)
        output_path = Path(output_path)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise VideoError(f"Could not open video: {input_path}")

        smoother = BoxSmoother(
            padding=self._config.padding,
            alpha=self._config.alpha,
            jump_threshold=self._config.jump_threshold,
        )

        writer: cv2.VideoWriter | None = None

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = 0

            for frame in self._iter_frames(cap, total_frames):
                detections = self._detector.detect(frame, self._config.conf)
                primary = select_primary_subject(
                    detections,
                    frame.shape[1],
                    frame.shape[0],
                    self._detector.track_ages,
                    self._config.track_id,
                )
                crop_box = smoother.update(primary, frame.shape[1], frame.shape[0])
                cropped = crop_frame(frame, crop_box, self._config.width, self._config.height)

                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # pyright: ignore[reportAttributeAccessIssue]
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(output_path),
                        fourcc,
                        fps,
                        (self._config.width, self._config.height),
                    )

                writer.write(cropped)

                if self._config.show:
                    preview = draw_detections(frame, detections)
                    cv2.imshow("Portal - Detection", preview)
                    cv2.imshow("Portal - Cropped", cropped)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_count += 1
                yield frame_count, total_frames

        finally:
            cap.release()
            cv2.destroyAllWindows()
            if writer is not None:
                writer.release()

    def _iter_frames(
        self, cap: cv2.VideoCapture, total_frames: int
    ) -> Generator[np.ndarray, None, None]:
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if self._config.skip > 0 and frame_index % (self._config.skip + 1) != 0:
                frame_index += 1
                continue
            frame_index += 1
            yield frame


class LiveProcessor:
    def __init__(self, detector: PersonDetector, config: ProcessorConfig) -> None:
        self._detector = detector
        self._config = config
        self._running = False

    def run(self, camera_id: int = 0, output_path: Path | None = None) -> None:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise VideoError(f"Could not open camera {camera_id}")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30

        capture_q: queue.Queue = queue.Queue(maxsize=1)
        result_q: queue.Queue = queue.Queue(maxsize=1)

        self._running = True
        frame_index = 0

        def capture_loop() -> None:
            nonlocal frame_index
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    self._running = False
                    break
                try:
                    capture_q.put_nowait((frame_index, frame))
                    frame_index += 1
                except queue.Full:
                    pass

        def inference_loop() -> None:
            smoother = BoxSmoother(
                padding=self._config.padding,
                alpha=self._config.alpha,
                jump_threshold=self._config.jump_threshold,
            )
            last_crop_box: tuple[int, int, int, int] | None = None
            last_detections: Sequence[Detection] = []

            while self._running:
                try:
                    fidx, frame = capture_q.get(timeout=1.0)
                except queue.Empty:
                    continue

                if self._config.skip == 0 or fidx % (self._config.skip + 1) == 0:
                    detections = self._detector.detect(frame, self._config.conf)
                    last_detections = detections
                    primary = select_primary_subject(
                        detections,
                        frame.shape[1],
                        frame.shape[0],
                        self._detector.track_ages,
                        self._config.track_id,
                    )
                    crop_box = smoother.update(primary, frame.shape[1], frame.shape[0])
                    last_crop_box = crop_box
                else:
                    detections = last_detections
                    crop_box = (
                        last_crop_box
                        if last_crop_box is not None
                        else (0, 0, frame.shape[1], frame.shape[0])
                    )

                cropped = crop_frame(frame, crop_box, self._config.width, self._config.height)

                try:
                    result_q.put_nowait((cropped, frame, detections))
                except queue.Full:
                    pass

        def display_loop() -> None:
            writer: cv2.VideoWriter | None = None

            if output_path is not None:
                fourcc = cv2.VideoWriter_fourcc(*"avc1")  # pyright: ignore[reportAttributeAccessIssue]
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    fps,
                    (self._config.width, self._config.height),
                )

            try:
                while self._running:
                    try:
                        cropped, original, detections = result_q.get(timeout=1.0)  # pyright: ignore[reportAssignmentType]
                    except queue.Empty:
                        continue

                    cv2.imshow("Portal - Original", draw_detections(original, detections))
                    cv2.imshow("Portal - Cropped", cropped)

                    if writer is not None:
                        writer.write(cropped)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self._running = False
                        break
            finally:
                cv2.destroyAllWindows()
                if writer is not None:
                    writer.release()

        capture_thread = threading.Thread(target=capture_loop, daemon=True)
        inference_thread = threading.Thread(target=inference_loop, daemon=True)

        capture_thread.start()
        inference_thread.start()

        try:
            display_loop()
        finally:
            self._running = False
            capture_thread.join(timeout=2)
            inference_thread.join(timeout=2)
            cap.release()
