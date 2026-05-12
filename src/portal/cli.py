import logging
from pathlib import Path

import typer

from portal.config import ProcessorConfig
from portal.detector import PersonDetector
from portal.processor import FileProcessor, LiveProcessor

logger = logging.getLogger("portal")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


app = typer.Typer(help="Portal - automatic video cropping around detected people")


def _build_config(
    model: str,
    conf: float,
    padding: float,
    width: int,
    height: int,
    alpha: float,
    jump_threshold: float,
    track_id: int | None,
    show: bool,
    skip: int,
    output: str | None = None,
) -> ProcessorConfig:
    return ProcessorConfig(
        model=model,
        conf=conf,
        padding=padding,
        width=width,
        height=height,
        alpha=alpha,
        jump_threshold=jump_threshold,
        track_id=track_id,
        show=show,
        skip=skip,
        output=output,
    )


@app.command()
def process(
    input: Path = typer.Argument(..., help="Input video file path", exists=True),
    output: Path = typer.Argument(..., help="Output video file path"),
    model: str = typer.Option("yolov8n.pt", help="YOLO model name"),
    conf: float = typer.Option(0.5, min=0.0, max=1.0, help="Detection confidence threshold"),
    padding: float = typer.Option(0.10, min=0.0, max=1.0, help="Padding around detected people"),
    width: int = typer.Option(1280, min=64, help="Output width"),
    height: int = typer.Option(720, min=64, help="Output height"),
    alpha: float = typer.Option(0.10, min=0.0, max=1.0, help="EMA smoothing factor"),
    jump_threshold: float = typer.Option(
        0.15,
        min=0.0,
        max=1.0,
        help="Fraction of frame diagonal for snap override",
    ),
    track_id: int | None = typer.Option(None, "--track-id", help="Lock to a specific track ID"),
    show: bool = typer.Option(False, "--show", help="Show preview windows"),
    skip: int = typer.Option(0, help="Process every Nth frame (0 = all frames)"),
) -> None:
    config = _build_config(
        model=model,
        conf=conf,
        padding=padding,
        width=width,
        height=height,
        alpha=alpha,
        jump_threshold=jump_threshold,
        track_id=track_id,
        show=show,
        skip=skip,
    )

    _setup_logging()

    detector = PersonDetector(config.model)
    logger.info("Loading model: %s (device: %s)", config.model, detector.device)
    detector.warmup()
    logger.info("Model ready")

    processor = FileProcessor(detector, config)

    with typer.progressbar(length=1, label="Processing") as progress:
        for current, total in processor.process_file(input, output):
            progress.label = f"Frame {current}/{total}"
            progress.update(1)

    logger.info("Done. Output written to %s", output)


@app.command()
def live(
    camera_id: int = typer.Argument(0, help="Camera device ID"),
    model: str = typer.Option("yolov8n.pt", help="YOLO model name"),
    conf: float = typer.Option(0.5, min=0.0, max=1.0, help="Detection confidence threshold"),
    padding: float = typer.Option(0.10, min=0.0, max=1.0, help="Padding around detected people"),
    width: int = typer.Option(1280, min=64, help="Output width"),
    height: int = typer.Option(720, min=64, help="Output height"),
    alpha: float = typer.Option(0.10, min=0.0, max=1.0, help="EMA smoothing factor"),
    jump_threshold: float = typer.Option(
        0.15,
        min=0.0,
        max=1.0,
        help="Fraction of frame diagonal for snap override",
    ),
    track_id: int | None = typer.Option(None, "--track-id", help="Lock to a specific track ID"),
    output: Path | None = typer.Option(None, help="Record cropped output to file"),
) -> None:
    config = _build_config(
        model=model,
        conf=conf,
        padding=padding,
        width=width,
        height=height,
        alpha=alpha,
        jump_threshold=jump_threshold,
        track_id=track_id,
        show=True,
        skip=0,
        output=str(output) if output else None,
    )

    _setup_logging()

    detector = PersonDetector(config.model)
    logger.info("Loading model: %s (device: %s)", config.model, detector.device)
    detector.warmup()
    logger.info("Starting camera %s...", camera_id)

    processor = LiveProcessor(detector, config)
    processor.run(camera_id, output)
