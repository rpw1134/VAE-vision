# MediaPipe Hand Detection

## What It Is

MediaPipe's Hand Landmarker is a pre-trained model that takes an image and returns the 3D positions of 21 keypoints on a hand (fingertips, knuckles, wrist). It also gives you a bounding box around the detected hand.

There are two APIs in the MediaPipe Python package — the legacy `mp.solutions.hands` and the newer **Tasks API** (`mediapipe.tasks.vision`). This project uses the Tasks API, which is more explicit about model loading and image formats.

---

## The Tasks API

### Model File

The Tasks API requires a `.task` model file downloaded separately — it is not bundled with the pip package.

- Download: `hand_landmarker.task` from the MediaPipe Models page
- Docs: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

### Core Classes

```python
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
```

| Class | Role |
|---|---|
| `HandLandmarkerOptions` | Configuration (model path, mode, num hands, thresholds) |
| `HandLandmarker` | The detector — call `.detect()` on each frame |
| `HandLandmarkerResult` | What `.detect()` returns |

### Setup

```python
options = mp_vision.HandLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=mp_vision.RunningMode.IMAGE,  # vs VIDEO or LIVE_STREAM
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = mp_vision.HandLandmarker.create_from_options(options)
```

`RunningMode.IMAGE` processes frames independently. For a webcam loop this is simplest — no timestamp bookkeeping.

### Running Detection

MediaPipe Tasks expects its own image wrapper, not a raw numpy array:

```python
import mediapipe as mp

mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
result = detector.detect(mp_image)
```

The input **must be RGB** (not BGR). OpenCV captures BGR by default, so conversion is necessary before wrapping.

---

## The Result Object

`result` is a `HandLandmarkerResult` with two main fields:

### `result.hand_landmarks`

A list (one per detected hand) of 21 `NormalizedLandmark` objects. Each landmark has:

| Field | Type | Range |
|---|---|---|
| `x` | float | 0.0 – 1.0 (fraction of image width) |
| `y` | float | 0.0 – 1.0 (fraction of image height) |
| `z` | float | depth relative to wrist (less reliable) |

To get pixel coordinates:
```python
px = int(landmark.x * frame_width)
py = int(landmark.y * frame_height)
```

### `result.handedness`

A list of classification results indicating `"Left"` or `"Right"` for each detected hand. Useful later but not needed for basic detection.

### Checking for No Detection

```python
if not result.hand_landmarks:
    # no hand in frame — yield original frame unchanged
    ...
```

---

## Landmark Indices

The 21 landmarks follow a fixed numbering scheme:

```
0  — WRIST
1  — THUMB_CMC
2  — THUMB_MCP
3  — THUMB_IP
4  — THUMB_TIP
5  — INDEX_FINGER_MCP
...
8  — INDEX_FINGER_TIP
...
12 — MIDDLE_FINGER_TIP
...
16 — RING_FINGER_TIP
...
20 — PINKY_TIP
```

Full diagram: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models

All 21 points are used together to compute the bounding box and convex hull mask — individual indices matter more for gesture recognition than for this project.

---

## Running Mode Trade-offs

| Mode | Use case | Notes |
|---|---|---|
| `IMAGE` | Independent frames | Simplest; no state between frames |
| `VIDEO` | Frame sequences with timestamps | Better tracking continuity |
| `LIVE_STREAM` | Async callback-based | Non-blocking; requires callback fn |

For an initial implementation `IMAGE` mode is fine. Switching to `VIDEO` mode later can improve landmark stability across frames.

---

## Relevant Links

- MediaPipe Hand Landmarker guide: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
- Landmark topology image: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models
- Tasks API Python reference: https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarker
