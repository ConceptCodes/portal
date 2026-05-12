from dataclasses import dataclass


@dataclass
class ProcessorConfig:
    model: str = "yolov8n.pt"
    conf: float = 0.5
    padding: float = 0.10
    width: int = 1280
    height: int = 720
    alpha: float = 0.10
    jump_threshold: float = 0.15
    track_id: int | None = None
    skip: int = 0
    show: bool = False
    output: str | None = None
