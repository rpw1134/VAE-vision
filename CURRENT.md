# VAE-Vision — Current State

## What It Does
Applies a real-time "ghostly hand" effect to webcam video. The hand is detected, its crop is encoded and decoded through a VAE, and the reconstruction is alpha-blended back over the live feed as a translucent echo of itself.

---

## Phase Status

### 1. Data Collection — DONE
- `data.py` captures webcam frames, detects a hand each frame, crops + resizes to 128×128, and accumulates into a NumPy array `(N, 128, 128, 3)` uint8
- Saves to `data/hands.npy` as a single file
- `BBOX_PADDING = 35` used for crop context
- ~2000 samples collected

### 2. Detection & Cropping — DONE
- `pipeline.py` — `build_detector()` loads MediaPipe HandLandmarker from `hand_landmarker.task`
- `detect_hand(frame, detector)` converts BGR→RGB, runs detection, returns a typed `HandDetection` dict with `detected`, `landmarks` (21 pixel-coord points), `bbox`, and `handedness`
- `utils.py` — `bgr_to_rgb` / `rgb_to_bgr` helpers
- `hand_types.py` — `Landmark`, `BBox`, `HandDetection` TypedDicts for linting

### 3. Model — DONE (trained)
- `model.py` — `VAE` composed of `Encoder` + `Decoder`
  - Encoder: 4× Conv2d with stride 2 (128→64→32→16→8), flatten, two linear heads for `mu` and `log_var`
  - Decoder: linear → reshape → 4× ConvTranspose2d (8→16→32→64→128), Sigmoid output
  - Latent dim: 128
- `training.py` — full training loop with:
  - `HyperParams` dataclass (lr, weight decay, batch size, epochs, beta schedule, scheduler knobs, val split)
  - AdamW optimizer, ReduceLROnPlateau scheduler
  - KL beta annealing: 0.0 → 1.0 over 20 epochs
  - 10% val split, pin memory, DataParallel for multi-GPU
  - Saves best checkpoint to `checkpoints/vae_best.pt` (by val loss)
- Trained on 2× T4 GPUs (Kaggle), ~1 minute, 100 epochs
- Final val loss: ~29,870 (~0.60 BCE per pixel)

**Model quality observations:**
- Conditioned reconstructions (encoder → reparameterize → decode): accurate color, tone, and position with slight blur — ideal for the ghost effect
- Prior samples (z ~ N(0,1) → decode): ~60-70% recognizable hands, remainder blurry or distorted — latent space partially regularized but not tight
- Unconditioned quality doesn't matter for the runtime pipeline, which always conditions on the live crop

### 4. Mask Generation — IN PROGRESS
- `mask.py` — `draw_debug()` renders landmarks (white dots), bbox (green), and convex hull (red) on a frame for visual verification
- Convex hull from 21 landmarks computed via `cv2.convexHull` — hull does not hug the palm perfectly, but Gaussian blur feathering will extend the ghost naturally past the hull boundary
- Soft mask with `cv2.GaussianBlur` not yet implemented

### 5. Blend & Display — NOT STARTED
- Alpha-blend decoded VAE output over original frame using soft mask
- `main.py` runtime loop not yet wired up

---

## File Responsibilities

| File | Status | What it owns |
|---|---|---|
| `model.py` | Done | VAE architecture — encoder, decoder, reparameterization |
| `training.py` | Done | Training loop, HyperParams, HandDataset |
| `data.py` | Done | Image collection loop, npy save, visualizer |
| `pipeline.py` | Done | MediaPipe detection, BGR→RGB, HandDetection dict |
| `hand_types.py` | Done | TypedDicts: Landmark, BBox, HandDetection |
| `utils.py` | Done | bgr_to_rgb, rgb_to_bgr |
| `mask.py` | Partial | Debug overlay done; soft mask not yet written |
| `exploration.py` | Done | Webcam loop, reconstruction viz, latent walk, prior sampling |
| `main.py` | Not started | Runtime webcam loop with full ghost pipeline |

---

## What's Left

1. **Soft mask** (`mask.py`) — fill convex hull on blank canvas, apply GaussianBlur to feather edges, return float32 `(H, W, 1)` mask
2. **Blend** (`pipeline.py`) — resize decoded output to bbox dims, alpha-blend with mask into original frame
3. **Runtime loop** (`main.py`) — load checkpoint, run full per-frame pipeline, display with `cv2.imshow`

---

## Tunable Knobs

- **`GHOST_ALPHA`** — decoded blend weight (0 = invisible, 1 = full replacement)
- **`LATENT_DIM`** — 128 currently; lower = ghostier, higher = sharper
- **`BLUR_KERNEL_SIZE`** — GaussianBlur kernel for mask feathering (larger = softer edges)
- **`BBOX_PADDING`** — pixels of context around hand crop (currently 35)
- **`beta_end`** — raise to 2.0–4.0 + shorten warmup to tighten prior if more unconditioned quality is needed
