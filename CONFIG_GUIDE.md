# Configuration Guide

This guide explains every parameter in the pipeline YAML config files.  
Config files live in the `configs/` folder and are passed via `--config`:

```bash
venv/bin/python main.py --config configs/rers2_config.yaml --input video.mp4 --output output_3
```

Two presets are provided:

| File | Purpose |
|---|---|
| `default_config.yaml` | Strict: high-confidence interactions only |
| `rers2_config.yaml` | Relaxed: suitable for most industrial CCTV videos |

---

## Top-Level Sections

```
video       → Input/output paths
device      → GPU settings
detection   → YOLOv8 detector
tracking    → ByteTrack tracker
pose        → ViTPose pose estimator
depth       → MiDaS depth estimator
interaction → Interaction scoring weights & thresholds
clip_mining → Clip extraction settings  ← most commonly tuned
logging     → Log level and file
performance → Speed vs accuracy trade-offs
```

---

## `video` — Input / Output

```yaml
video:
  input_path: ""          # (optional) default video path; overridden by --input CLI arg
  output_dir: "output"    # (optional) default output folder; overridden by --output CLI arg
  fps: null               # null = use video's native FPS
```

> You almost never need to change this section — use CLI args instead.

---

## `device` — GPU Settings

```yaml
device:
  use_gpu: true       # true = auto-detect GPU; false = force CPU
  gpu_id: 0          # which GPU to use (0 = first GPU)
  fallback_to_cpu: true  # if GPU not found, continue on CPU instead of crashing
```

| Parameter | Effect |
|---|---|
| `use_gpu: false` | Forces CPU mode (very slow for long videos) |
| `gpu_id: 1` | Use second GPU if multiple GPUs are present |

---

## `detection` — YOLOv8 Object Detector

```yaml
detection:
  model: "yolov8x.pt"     # model size: n (fastest) → s → m → l → x (most accurate)
  img_size: 1280          # inference resolution in pixels
  conf_threshold: 0.25    # minimum detection confidence (0–1)
  iou_threshold: 0.45     # NMS IOU for removing duplicate boxes
  person_class: 0         # COCO class ID for person (do not change)
  machine_classes: [2, 3, 5, 7, 56, 57, 58, 59, 60, 61, 62, 63,
                    64, 65, 66, 67, 73, 74, 75, 76, 77, 78, 79]
  batch_size: 8
```

### Model size trade-offs

| Model | Speed | Accuracy | GPU Memory |
|---|---|---|---|
| `yolov8n.pt` | Fastest | Lowest | ~2 GB |
| `yolov8m.pt` | Balanced | Good | ~4 GB |
| `yolov8x.pt` | Slowest | Best | ~8 GB |

### `machine_classes` — COCO class IDs

These are the COCO dataset class IDs treated as "machines/equipment".  
Common IDs:

| ID | Object |
|---|---|
| 2 | car |
| 3 | motorcycle |
| 5 | bus |
| 7 | truck |
| 56 | chair |
| 57–67 | tables, monitors, keyboards, mice, remotes |
| 73–79 | laptops, TVs, ovens, fridges, etc. |

> **Tip**: If the pipeline detects no machines, your equipment may not match these COCO classes. Consider lowering `conf_threshold` to `0.15`.

### Key tuning parameters

| Parameter | Lower → | Higher → |
|---|---|---|
| `conf_threshold` | More detections (more noise) | Fewer detections (misses objects) |
| `img_size` | Faster, misses small objects | Slower, catches small objects |

---

## `tracking` — ByteTrack

```yaml
tracking:
  track_thresh: 0.5     # minimum score for a detection to start a track
  track_buffer: 30      # frames to keep a lost track alive before deleting
  match_thresh: 0.8     # IOU threshold for matching detections to existing tracks
  min_box_area: 100     # minimum bounding box area in pixels (filters tiny noise)
  mot20: false          # use MOT20 benchmark settings (leave false)
```

| Parameter | Guidance |
|---|---|
| `track_buffer` | Increase if people/machines frequently go off-screen briefly |
| `min_box_area` | Increase to filter very small, noisy detections |
| `match_thresh` | Lower if tracking IDs switch unexpectedly between frames |

---

## `pose` — ViTPose Pose Estimator

```yaml
pose:
  model: "vitpose-b"        # model size: b (base), l (large), h (huge)
  conf_threshold: 0.3       # minimum keypoint confidence
  batch_size: 16
  hand_keypoints: [9, 10]   # COCO keypoints: left wrist (9), right wrist (10)
  arm_keypoints: [5, 6, 7, 8, 9, 10]  # shoulders, elbows, wrists
```

> **Note**: MMPose is required for full ViTPose support. Without it, a simplified pose estimator is used automatically.

The 17 COCO keypoints (for reference):

```
0: nose        5: left_shoulder   10: right_wrist
1: left_eye    6: right_shoulder  11: left_hip
2: right_eye   7: left_elbow      12: right_hip
3: left_ear    8: right_elbow     13: left_knee
4: right_ear   9: left_wrist      14: right_knee
                                  15: left_ankle
                                  16: right_ankle
```

---

## `depth` — MiDaS Depth Estimator

```yaml
depth:
  model: "DPT_Large"   # DPT_Large (best), DPT_Hybrid (balanced), MiDaS_small (fastest)
  optimize: true        # enable half-precision optimization on GPU
  height: 384           # input height for depth model
  square: false         # false = keep aspect ratio; true = force square crop
```

| Model | Speed | Quality |
|---|---|---|
| `MiDaS_small` | Fastest | Lower quality depth |
| `DPT_Hybrid` | Balanced | Good |
| `DPT_Large` | Slowest | Best depth accuracy |

---

## `interaction` — Interaction Scoring

This section controls **how** person-machine interactions are scored (0–1) per frame.  
The final score is a weighted sum of four components.

```yaml
interaction:
  proximity:
    enabled: true
    max_distance: 3.0   # max normalized depth distance to count as "near"
    weight: 0.30        # contribution to final score

  pose:
    enabled: true
    weight: 0.25
    reaching_threshold: 0.25   # how far a hand must extend toward machine
    operating_threshold: 0.40  # how close a hand must be to machine

  motion:
    enabled: true
    weight: 0.20
    temporal_window: 15   # frames to compute motion over
    min_movement: 3       # minimum pixel movement to register as active motion

  hoi:
    enabled: false   # Human-Object Interaction model (requires extra setup)
    weight: 0.25

  idle_detection:
    enabled: true
    standing_threshold: 5.0    # max movement (px) to classify person as idle/standing
    walking_speed_max: 60.0    # max speed (px/frame) before person is considered walking away
```

### Score weights must reflect priorities

The weights for `proximity`, `pose`, `motion`, and `hoi` should be balanced for your scenario:

| Scenario | Recommendation |
|---|---|
| Proximity is most reliable | Increase `proximity.weight` to `0.5` |
| Lots of hand-tool work | Increase `pose.weight` |
| Workers are always near machines | Rely more on `motion` and `pose` |

### Idle detection

When `idle_detection` is enabled, persons who are standing still or just walking through are penalized, reducing false-positive clip extraction.

- Raise `standing_threshold` if workers work at fixed stations (they may be legitimately still while interacting)
- Raise `walking_speed_max` if workers move quickly between machines

---

## `clip_mining` — Clip Extraction ← **Most Important Section**

```yaml
clip_mining:
  clip_duration: 10.0          # length of each output clip in seconds
  sliding_window: 5.0          # how much to advance the window each step (seconds)
  interaction_threshold: 0.4   # minimum per-frame score to count as "interacting"
  min_interaction_frames: 40   # minimum frames above threshold needed to save a clip
  nms_threshold: 0.1           # IOU threshold for removing overlapping clips
  save_metadata: true          # save clip_XXXX.json alongside each clip
  save_visualization: true     # save annotated clips with bounding boxes and scores
```

### Key parameters explained

#### `interaction_threshold`
Frames with score ≥ this value are counted as "active interaction" frames.

| Value | Effect |
|---|---|
| `0.6` (strict) | Only very obvious, close interactions captured |
| `0.4` (relaxed) | Catches more subtle interactions |
| `0.3` (very relaxed) | May include false positives |

#### `min_interaction_frames`
A sliding window is only saved as a clip if at least this many frames are above `interaction_threshold`.

| Value | Effect |
|---|---|
| `60` | Requires 6 seconds of interaction in a 10s clip at 10fps |
| `40` | Requires 4 seconds — more lenient |

#### `nms_threshold` — Controls clip overlap
After scoring, NMS removes clips that overlap temporally with a higher-scoring clip.

| Value | Effect |
|---|---|
| `0.1` | Clips with >10% time overlap are removed — **prevents overlapping clips** |
| `0.5` | Clips can share up to 50% of their duration — may produce overlapping clips |

> **Recommendation**: Keep `nms_threshold` at `0.1` to avoid duplicate overlapping clips.

#### `save_visualization`
When `true`, extracted clips are annotated with bounding boxes, track IDs, pose keypoints, and interaction scores.  
Set to `false` for faster extraction if you only need the raw clips.

---

## `logging` — Logging

```yaml
logging:
  level: "INFO"          # DEBUG / INFO / WARNING / ERROR
  save_logs: true        # write logs to file
  log_file: "pipeline.log"
  progress_bar: true     # show tqdm progress bar during processing
```

Set `level: "DEBUG"` to see per-frame detection and scoring details.

---

## `performance` — Speed vs. Accuracy

```yaml
performance:
  frame_skip: 1      # process every Nth frame (1 = all, 2 = every other, etc.)
  max_frames: null   # stop after N frames (null = entire video)
  num_workers: 4     # parallel data loading workers
```

| Parameter | Guidance |
|---|---|
| `frame_skip: 2` | ~2× speed, may miss brief interactions |
| `max_frames: 500` | Quick test run without processing the full video |

---

## Preset Comparison: `default_config` vs `rers2_config`

| Parameter | `default_config.yaml` | `rers2_config.yaml` |
|---|---|---|
| `interaction.proximity.max_distance` | 2.0 | **3.0** |
| `interaction.pose.reaching_threshold` | 0.3 | **0.25** |
| `interaction.pose.operating_threshold` | 0.5 | **0.40** |
| `interaction.motion.min_movement` | 5 | **3** |
| `interaction.idle_detection.standing_threshold` | 3.0 | **5.0** |
| `interaction.idle_detection.walking_speed_max` | 50.0 | **60.0** |
| `clip_mining.interaction_threshold` | 0.6 | **0.4** |
| `clip_mining.min_interaction_frames` | 60 | **40** |
| `clip_mining.nms_threshold` | 0.1 | 0.5 |

> Use `rers2_config.yaml` as your starting point for new industrial videos. Tighten thresholds gradually if you get too many false-positive clips.

---

## Creating a Custom Config

1. Copy an existing preset:
   ```bash
   cp configs/rers2_config.yaml configs/my_config.yaml
   ```
2. Edit the parameters you want to change.
3. Run with your new config:
   ```bash
   venv/bin/python main.py --config configs/my_config.yaml --input video.mp4 --output my_output
   ```
