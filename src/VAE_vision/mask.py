import cv2
import numpy as np

from VAE_vision.hand_types import HandDetection


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
