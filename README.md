# Portal

Autonomous video cropping — detect people in frame and crop to focus on them. Designed for live performances (church services, conferences, talks) where you want an auto-PTZ effect without a camera operator.

## Features

- **Person detection** via YOLO + ByteTrack with stable track IDs
- **Primary subject selection** using composite scoring (size × centrality × age)
- **Adaptive EMA smoothing** with jump detection for cinematic pan-only motion
- **30-frame coast** on lost detections — no jarring zoom-outs from brief occlusion
- **Fixed output resolution** with preallocated resize buffer — no per-frame allocation
- **Live mode** with 3-thread pipeline (capture, inference, display) and drop queues for real-time operation
- **File mode** for batch processing pre-recorded videos
- **MPS (Apple Silicon)**, CUDA, and CPU support
- **Manual track lock** — pin the crop to a specific person by track ID

## Installation

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

## Usage

### Live camera

```bash
portal live
```

With options:

```bash
portal live 0 --model yolov8n.pt --width 1280 --height 720 --alpha 0.1 --output recording.mp4
```

### Process a video file

```bash
portal process input.mp4 output.mp4
```

With options:

```bash
portal process input.mp4 output.mp4 --width 1920 --height 1080 --track-id 3 --show
```

### CLI Options

| Option | Default | Description |
|---|---|---|
| `--model` | `yolov8n.pt` | YOLO model name |
| `--conf` | `0.5` | Detection confidence threshold (0–1) |
| `--padding` | `0.10` | Padding fraction around detected people (0–1) |
| `--width` | `1280` | Output width |
| `--height` | `720` | Output height |
| `--alpha` | `0.10` | EMA smoothing factor (0–1) |
| `--jump-threshold` | `0.15` | Frame diagonal fraction for snap override (0–1) |
| `--track-id` | — | Lock to a specific track ID |
| `--show` | — | Show preview windows (file mode) |
| `--skip` | `0` | Process every Nth frame (0 = all) |
| `--output` | — | Record live feed to file (live mode) |

## How It Works

### Pipeline (Live Mode)

```
[Camera] → [Capture Thread] → capture_q(maxsize=1, drop) → [Inference Thread] → result_q(maxsize=1, drop) → [Main Thread: Display + Record]
```

Both queues use `maxsize=1` with drop-on-full — frames are discarded when the downstream consumer is busy. Recency > completeness.

### Primary Subject Selection

When multiple people are detected, a composite score selects the subject:

- **50%** — Box area (closest/largest person dominates)
- **35%** — Center distance (broadcasting convention)
- **15%** — Track age (prevents flickering to new entrants)

### Crop Smoothing

- **Normal motion**: EMA with α=0.10 (~30-frame lag, cinematic smoothness)
- **Jump detected** (subject moved >15% of frame diagonal): α=1.0, snap immediately
- **Pan-only**: center updates every frame, size only when area changes >20%

## Project Structure

```
src/portal/
├── __init__.py      # Version
├── __main__.py      # python -m portal entry point
├── cli.py           # Typer CLI
├── config.py        # ProcessorConfig dataclass
├── detector.py      # PersonDetector: YOLO + ByteTrack
├── cropper.py       # Subject selection, BoxSmoother, crop, drawing
└── processor.py     # FileProcessor (sequential), LiveProcessor (3-thread)
```

## Development

```bash
uv sync --group dev
ruff check src/portal/ tests/
ruff format src/portal/ tests/
pyright src/portal/
pytest tests/
```

## License

MIT
