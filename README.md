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
- **Output sinks**: display preview, record to file, broadcast via NDI, stream to RTMP/RTSP/UDP — all simultaneously
- **MPS (Apple Silicon)**, CUDA, and CPU support
- **Manual track lock** — pin the crop to a specific person by track ID

## Installation

```bash
uv sync
```

## Usage

### Live camera

```bash
portal live
```

Record to file, broadcast via NDI, and stream to RTMP — all at once:

```bash
portal live 0 \
  --output service.mp4 \
  --ndi-name "Portal Cam" \
  --stream-url rtmp://church.tv/live/stream
```

Headless mode (no preview windows, e.g. on a server):

```bash
portal live --no-display --ndi-name "Portal Cam" --output recording.mp4
```

### Process a video file

```bash
portal process input.mp4 output.mp4
```

```bash
portal process input.mp4 output.mp4 --width 1920 --height 1080 --track-id 3 --show
```

### CLI Reference

#### `portal live`

| Option | Default | Description |
|---|---|---|
| `camera_id` | `0` | Camera device ID |
| `--model` | `yolov8n.pt` | YOLO model name |
| `--conf` | `0.5` | Detection confidence threshold (0–1) |
| `--padding` | `0.10` | Padding fraction around detected people (0–1) |
| `--width` | `1280` | Output width |
| `--height` | `720` | Output height |
| `--alpha` | `0.10` | EMA smoothing factor (0–1) |
| `--jump-threshold` | `0.15` | Frame diagonal fraction for snap override (0–1) |
| `--track-id` | — | Lock to a specific track ID |
| `--output` | — | Record cropped feed to file |
| `--ndi-name` | — | Broadcast cropped feed as an NDI source |
| `--stream-url` | — | Stream to URL (e.g. `rtmp://…`, `udp://…`, `srt://…`) |
| `--no-display` | — | Disable preview windows (headless mode) |

#### `portal process`

| Option | Default | Description |
|---|---|---|
| `input` | — | Input video file path |
| `output` | — | Output video file path |
| `--model` | `yolov8n.pt` | YOLO model name |
| `--conf` | `0.5` | Detection confidence threshold (0–1) |
| `--padding` | `0.10` | Padding fraction around detected people (0–1) |
| `--width` | `1280` | Output width |
| `--height` | `720` | Output height |
| `--alpha` | `0.10` | EMA smoothing factor (0–1) |
| `--jump-threshold` | `0.15` | Frame diagonal fraction for snap override (0–1) |
| `--track-id` | — | Lock to a specific track ID |
| `--show` | — | Show preview windows |
| `--skip` | `0` | Process every Nth frame (0 = all) |

## Output Sinks (Live Mode)

All sinks can be combined freely. The live mode fans out the cropped feed to every enabled sink.

| Sink | Class | How it works |
|---|---|---|
| **Display** | `DisplaySink` | Shows cropped feed in an OpenCV window. Enabled by default; disable with `--no-display`. |
| **File** | `FileSink` | Writes to `.mp4` using `cv2.VideoWriter` with `avc1` encoding. Enabled with `--output path.mp4`. |
| **NDI** | `NDISink` | Broadcasts as an NDI source on the local network. OBS, vMix, hardware switchers, and any NDI-compatible system can consume it. Requires `ndi-python` + NDI SDK (see [NDI SDK](https://ndi.video/tools/ndi-sdk/)). Enabled with `--ndi-name "Name"`. |
| **Stream** | `StreamSink` | Pipes raw frames to an `ffmpeg` subprocess for encoding and delivery. Supports RTMP, RTSP, UDP, SRT, and any ffmpeg-compatible output URL. Requires `ffmpeg` on `$PATH`. Enabled with `--stream-url <url>`. |

### Church production workflow

```
Camera → [Portal auto-crop] → NDI "Portal Cam" → OBS/vMix/switcher → house monitors + live stream
                              ├── recording.mp4 (archive)
                              └── rtmp://youtube.com (backup stream)
```

## How It Works

### Pipeline (Live Mode)

```
[Camera] → [Capture Thread] → capture_q(maxsize=1, drop) → [Inference Thread] → result_q(maxsize=1, drop) → [Main Thread: Sink fan-out]
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
├── cropper.py       # Subject selection, BoxSmoother, crop, drawing
├── detector.py      # PersonDetector: YOLO + ByteTrack
├── processor.py     # FileProcessor (sequential), LiveProcessor (3-thread)
└── sinks.py         # VideoSink abstraction + Display/File/NDI/Stream/Composite sinks
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
