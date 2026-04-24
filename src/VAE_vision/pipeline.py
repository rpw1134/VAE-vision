import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from VAE_vision.hand_types import BBox, HandDetection, Landmark
from VAE_vision.utils import bgr_to_rgb

MODEL_PATH = "hand_landmarker.task"


def build_detector(model_path: str = MODEL_PATH) -> mp_vision.HandLandmarker:
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def detect_hand(frame: np.ndarray, detector: mp_vision.HandLandmarker) -> HandDetection:
    h, w = frame.shape[:2]

    rgb = bgr_to_rgb(frame)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return {"detected": False, "landmarks": [], "bbox": None, "handedness": None}

    raw_landmarks = result.hand_landmarks[0]
    landmarks = [
        {"x_px": int(lm.x * w), "y_px": int(lm.y * h), "z": lm.z}
        for lm in raw_landmarks
    ]

    xs = [lm["x_px"] for lm in landmarks]
    ys = [lm["y_px"] for lm in landmarks]
    bbox = {
        "x_min": min(xs),
        "y_min": min(ys),
        "x_max": max(xs),
        "y_max": max(ys),
    }

    handedness = result.handedness[0][0].display_name if result.handedness else None

    return {
        "detected": True,
        "landmarks": landmarks,
        "bbox": bbox,
        "handedness": handedness,
    }
