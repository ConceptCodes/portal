# Portal Architecture

Autonomous video cropping — detect people in frame, crop to focus on them. Designed for live performances (church services, conferences, talks) where you want an auto-PTZ effect without a camera operator.

---

## Core Invariant

> **Always work on the newest frame, never buffer old ones.**

This drives every architectural decision: drop-queues, single-element buffers, fixed output resolution, frame skipping.

---

## Pipeline (Live Mode)

```
[Camera] → [Capture Thread] → capture_q(maxsize=1, drop) → [Inference Thread] → result_q(maxsize=1, drop) → [Main Thread: Display + Record]
```

### Threads

| Thread | Job | Critical Details |
|---|---|---|
| **Capture** | `cap.read()` in loop | `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)` reduces OpenCV internal buffer; `put_nowait()` + drop if queue full |
| **Inference** | `model.track(persist=True)` + crop + smooth | Maintains `track_ages` dict, coast/freeze state, `BoxSmoother` state |
| **Main** | `cv2.imshow()` + `writer.write()` | All OpenCV GUI calls must be on the main thread (macOS requirement) |

### Queues

- Both queues: `queue.Queue(maxsize=1)`
- Capture writes via `put_nowait()` — drops frame if inference is still busy
- Inference writes via `put_nowait()` — drops result if display hasn't consumed previous one yet
- This is **not a bug** — it's the correct design for real-time. Recency > completeness.

---

## Person Detection

### Model

- **Default**: `yolov8n.pt` (3.2M params, ~6.5 MB disk, ~13 MB RAM)
- **MPS (Apple Silicon)**: ~117 fps (720p), requires explicit `device="mps"`
- **CPU only**: ~40 fps (720p), viable for 30fps input with headroom
- Device resolves at runtime: `"mps" if torch.backends.mps.is_available() else "cpu"`

### Tracker

- **ByteTrack** via `model.track(persist=True, tracker="bytetrack.yaml")`
- **Why tracking over detection**: ByteTrack maintains stable integer track IDs across frames, handles occlusions via two-stage confidence cascade (recovers low-confidence detections), and holds lost tracks for up to 30 frames via Kalman prediction
- **Without tracking**: Raw detection flickers — a person at `conf=0.49` disappears with `conf=0.5` threshold, causing jarring crop jumps
- **`persist=True` is mandatory** — without it, tracker state resets every frame

### Warmup

Two warmup calls on init to pre-compile Metal shaders (MPS) and avoid a 500ms+ first-frame spike:

```python
dummy = np.zeros((640, 640, 3), dtype=np.uint8)
model(dummy, device=device, verbose=False)  # shader compilation
model(dummy, device=device, verbose=False)  # kernel caching
```

---

## Primary Subject Selection

When multiple people are detected, a composite score chooses the subject:

```
score = 0.50 * area_ratio + 0.35 * center_score + 0.15 * age_ratio
```

| Component | Weight | Purpose |
|---|---|---|
| `box_area / frame_area` | 0.50 | Proximity = relevance. Closest/largest person dominates. |
| `1.0 - center_dist / max_dist` | 0.35 | Broadcasting convention — the speaker is framed centrally. |
| `track_age / max_age` | 0.15 | Prevents flickering to a newly-entered person who briefly scores higher. |

### Manual Lock (`--track-id`)

If `lock_track_id` is set and that ID is currently tracked, it is returned immediately—bypassing the composite score. The ID is displayed on the preview window so the user can identify which person to lock onto.

### No-Detection Handling

- Coast for up to 30 frames (~1 second at 30fps) using the last known crop position
- After 30 frames: graceful widen to full frame
- This avoids jarring zoom-outs from brief occlusion, side turns, or confidence dips

---

## Crop Box Smoothing

### Adaptive EMA

```
smoothed[t] = α * raw[t] + (1 - α) * smoothed[t-1]
```

- Normal motion: `α = 0.10` (~30-frame lag, cinematic smoothness)
- Jump detected (subject center moved >15% of frame diagonal in one frame): `α = 1.0` (snap immediately)
- Threshold detected via Euclidean distance of center normalized by frame diagonal

### Pan-Only Behavior

- Crop center `(cx, cy)` is smoothed every frame
- Crop size `(w, h)` only updates when detection area changes by >20%
- Simultaneously rezooming and panning looks amateur — this pattern is cinematic ("virtual PTZ")

### Fixed Output Resolution

- Configurable via `--width` and `--height` (default: 1280×720)
- Crop region is always resized to output resolution using a preallocated numpy buffer
- `cv2.resize(src, (w, h), dst=preallocated_buf)` — avoids per-frame allocation
- Fixed resolution is required because `cv2.VideoWriter` cannot accept variable-size frames

---

## Frame Skipping

| `--skip` | Inference rate | Best for |
|---|---|---|
| 0 | Every frame (30 fps inference) | MPS / GPU with headroom |
| 2 | Every 3rd frame (10 fps inference) | CPU-only fallback |
| 4 | Every 5th frame (6 fps inference) | Low-power / weak CPU |

- Skipped frames use the last inferred crop box (no interpolation for simplicity)
- The output framerate matches the input — skipped frames just reuse the last detection

---

## File Processing (Pre-recorded Videos)

`portal process` uses a sequential pipeline (no threading — for pre-recorded files, speed is bounded by encode/decode, not inference):

1. Open `cv2.VideoCapture(input)`
2. Loop frames: detect → select subject → smooth → resize → write
3. Progress reported via typer progress bar using `(current, total)` generator yields
4. `cv2.VideoWriter(release())` in `finally` block (critical for H.264 encoder buffer flush)

---

## Directory Structure

```
src/portal/
├── __init__.py         # __version__
├── __main__.py         # python -m portal
├── cli.py              # typer CLI: process + live commands
├── config.py           # ProcessorConfig dataclass
├── detector.py         # PersonDetector: YOLO + ByteTrack
├── cropper.py          # select_primary_subject, BoxSmoother, crop_frame, draw_detections
└── processor.py        # FileProcessor (sequential), LiveProcessor (3-thread)
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `opencv-python` | >=4.10 | Video I/O, frame manipulation, resize, draw |
| `ultralytics` | >=8.3 | YOLO model + ByteTrack |
| `numpy` | >=2.0 | Array operations, preallocated buffers |
| `typer` | >=0.15 | CLI framework |

Dev: `ruff`, `pytest`

---

## CLI Reference

### `portal process`

```bash
portal process INPUT OUTPUT [--model] [--conf] [--padding] [--width] [--height]
                           [--alpha] [--jump-threshold] [--track-id] [--show] [--skip]
```

### `portal live`

```bash
portal live [CAMERA_ID] [--model] [--conf] [--padding] [--width] [--height]
            [--alpha] [--jump-threshold] [--track-id] [--output]
```

- `CITYPEPE_ID`: Default 0. Integer camera device ID.
- `--model`: YOLO model name (default: `yolov8n.pt`).
- `--conf`: Detection confidence threshold 0.0–1.0 (default: 0.5).
- `--padding`: Padding fraction around detected people (default: 0.10).
- `--width`, `--height`: Output resolution (default: 1280×720).
- `--alpha`: EMA smoothing factor (default: 0.10).
- `--jump-threshold`: Fraction of frame diagonal for snap override (default: 0.15).
- `--track-id`: Lock to a specific track ID (displayed in preview).
- `--show`: Show preview windows (`process` only; `live` always shows).
- `--skip`: Process every Nth frame (default: 0 = all frames).
- `--output`: Record live feed to file (`live` only).

---

## Memory Safety Checklist

- [ ] `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)` — reduces OpenCV's internal frame buffer
- [ ] All queues use `maxsize=1` with drop-on-full — no unbounded growth
- [ ] `VideoWriter.release()` in `finally` block — flushes encoder buffer
- [ ] Preallocated numpy buffer for `cv2.resize(dst=...)` — avoids per-frame allocation
- [ ] `collections.deque(maxlen=N)` if any frame history is needed — not a raw list
- [ ] Daemon threads — die when main thread exits, no zombie threads
- [ ] No frame accumulation in lists — frames are processed and discarded
- [ ] MPS shares unified memory — total system RAM pressure includes model weights + activations + frames

---

## Future Considerations

- **Scene change detection:** If the frame content changes dramatically (different camera angle), reset the smoother and tracker
- **Multiple camera support:** N capture threads, M inference threads, one display thread — but needs a shot-switching heuristic
- **RTMP input/output:** Instead of cv2.VideoCapture, use ffmpeg subprocess piped to stdin/stdout for network streaming
- **GPU fallback:** CUDA path for NVIDIA GPUs — `device="cuda:0"` resolves trivially in ultralytics
