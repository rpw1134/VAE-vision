# Computer Vision: Bounding Boxes, Cropping, and Landmark Projection

## The Core Idea

Once MediaPipe gives you 21 normalized landmarks, the job is to:
1. Convert them to pixel coordinates
2. Compute a tight bounding box that encloses the whole hand
3. Add padding so the hand isn't right at the edge
4. Crop that region from the frame
5. Resize the crop to a fixed size for the VAE

These are all standard NumPy/OpenCV operations — no ML involved.

---

## Coordinate Systems

MediaPipe returns landmarks in **normalized image space**: `x` and `y` are fractions of the image dimensions, both in `[0.0, 1.0]`. You have to multiply by the frame's actual pixel dimensions to get usable coordinates.

```
pixel_x = landmark.x * frame_width
pixel_y = landmark.y * frame_height
```

Frame dimensions come from the NumPy array shape: `frame.shape` returns `(height, width, channels)` — note height comes first, opposite of the x/y convention.

---

## Building a Bounding Box from Landmarks

A bounding box is just the min and max x/y values across all landmark points:

```
x_min = min of all landmark pixel x values
x_max = max of all landmark pixel x values
y_min = min of all landmark pixel y values
y_max = max of all landmark pixel y values
```

This gives a tight box. In practice you add padding on all sides so the hand crop includes a bit of context and fingers aren't clipped.

**Padding + clamping to frame bounds:**

```
x_min = max(0, x_min - padding)
y_min = max(0, y_min - padding)
x_max = min(frame_width,  x_max + padding)
y_max = min(frame_height, y_max + padding)
```

Clamping is important — without it a hand near the edge of the frame produces a crop region that extends outside the image, which NumPy will either clip silently or raise an error on.

---

## Cropping a NumPy Array

OpenCV frames are NumPy arrays with shape `(H, W, C)`. Cropping is pure NumPy slicing — no OpenCV function needed:

```python
crop = frame[y_min:y_max, x_min:x_max]  # note: rows (y) first
```

The result is a view into the original array (not a copy), so modifications to `crop` affect `frame`. If you need an independent copy, use `.copy()`.

---

## Resizing to a Fixed Size

The VAE expects a fixed spatial input (e.g. 128×128). Use `cv2.resize`:

```python
resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
```

`cv2.resize` takes `(width, height)` — the opposite of NumPy's `(rows, cols)` convention. This is a common source of bugs when mixing the two.

Interpolation options:
| Flag | When to use |
|---|---|
| `INTER_LINEAR` | General upscaling/downscaling — good default |
| `INTER_AREA` | Shrinking — better quality when reducing significantly |
| `INTER_NEAREST` | Masks and label maps — preserves hard edges |

---

## Landmark Projection Into Crop Space

For the mask step (later), you need landmark coordinates relative to the crop, not the full frame. After cropping, subtract the crop's top-left corner:

```
local_x = pixel_x - x_min
local_y = pixel_y - y_min
```

Then scale by the resize factor if you resized:

```
scale_x = target_w / crop_w
scale_y = target_h / crop_h

scaled_x = local_x * scale_x
scaled_y = local_y * scale_y
```

This gives you landmark positions that map correctly onto the resized crop — needed for drawing the convex hull mask.

---

## BGR vs RGB — Where It Matters

| Operation | Color format |
|---|---|
| `cv2.VideoCapture` output | BGR |
| MediaPipe Tasks input | RGB |
| VAE training data (if loaded with PIL) | RGB |
| `cv2.imshow` input | BGR |

The practical rule: convert BGR→RGB before any MediaPipe call, and convert back for display. Everything in between (crop, resize, normalize) is color-format agnostic since it operates on all channels uniformly.

```python
rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
```

---

## Debugging Visually

Before wiring up the VAE, it's worth drawing detections on-screen to verify the pipeline works:

```python
# Draw bounding box
cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

# Draw each landmark
for lm in landmarks:
    px = int(lm.x * w)
    py = int(lm.y * h)
    cv2.circle(frame, (px, py), radius=4, color=(0, 0, 255), thickness=-1)
```

A green box + red dots on each hand confirms detection and coordinate math are correct before adding more complexity.

---

## Relevant OpenCV Docs

- `cv2.resize`: https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d
- `cv2.rectangle`: https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html
- `cv2.cvtColor`: https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html
- NumPy array indexing (slicing): https://numpy.org/doc/stable/user/basics.indexing.html
