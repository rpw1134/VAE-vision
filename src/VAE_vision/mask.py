import cv2
import numpy as np

from VAE_vision.hand_types import BBox, HandDetection, Landmark

BLUR_KERNEL_SIZE = 51


def build_soft_mask(
    landmarks: list[Landmark],
    bbox: BBox,
) -> np.ndarray:
    """
    Returns a float32 (H, W, 1) mask in [0, 1] for the crop defined by bbox.
    Convex hull of landmarks is filled and Gaussian-blurred to feather edges.
    """
    x_min, y_min = bbox["x_min"], bbox["y_min"]
    crop_h = bbox["y_max"] - y_min
    crop_w = bbox["x_max"] - x_min

    points = np.array(
        [[lm["x_px"] - x_min, lm["y_px"] - y_min] for lm in landmarks],
        dtype=np.int32,
    )
    points[:, 0] = np.clip(points[:, 0], 0, crop_w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, crop_h - 1)

    canvas = np.zeros((crop_h, crop_w), dtype=np.float32)
    hull = cv2.convexHull(points)
    cv2.fillPoly(canvas, [hull], 1.0)

    blurred = cv2.GaussianBlur(canvas, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)
    return blurred[:, :, np.newaxis]


def build_square_mask(bbox: BBox) -> np.ndarray:
    """
    Returns a float32 (H, W, 1) mask in [0, 1] for the crop defined by bbox.
    Full rectangle filled with 1.0 and Gaussian-blurred to feather edges.
    """
    crop_h = bbox["y_max"] - bbox["y_min"]
    crop_w = bbox["x_max"] - bbox["x_min"]
    canvas = np.ones((crop_h, crop_w), dtype=np.float32)
    blurred = cv2.GaussianBlur(canvas, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)
    return blurred[:, :, np.newaxis]


def draw_debug(frame: np.ndarray, detection: HandDetection) -> None:
    if not detection["detected"] or detection["bbox"] is None:
        return

    landmarks = detection["landmarks"]
    bbox = detection["bbox"]

    for lm in landmarks:
        cv2.circle(frame, (lm["x_px"], lm["y_px"]), radius=5, color=(255, 255, 255), thickness=3)

    cv2.rectangle(
        frame,
        (bbox["x_min"], bbox["y_min"]),
        (bbox["x_max"], bbox["y_max"]),
        color=(0, 255, 0),
        thickness=2,
    )

    points = np.array([[lm["x_px"], lm["y_px"]] for lm in landmarks], dtype=np.int32)
    hull = cv2.convexHull(points)
    cv2.polylines(frame, [hull], isClosed=True, color=(0, 0, 255), thickness=2)
