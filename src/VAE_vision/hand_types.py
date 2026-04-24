from typing import TypedDict


class Landmark(TypedDict):
    x_px: int
    y_px: int
    z: float


class BBox(TypedDict):
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class HandDetection(TypedDict):
    detected: bool
    landmarks: list[Landmark]
    bbox: BBox | None
    handedness: str | None
