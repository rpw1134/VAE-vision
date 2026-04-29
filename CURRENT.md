# VAE-Vision — Current State

## What It Does
Applies a real-time "ghostly hand" effect to webcam video. The hand is detected, its crop is encoded and decoded through a model (VAE or VQ-VAE), and the reconstruction is alpha-blended back over the live feed as a translucent echo of itself. The pipeline supports left hand (VAE), right hand (VQ-VAE), or both simultaneously.

---

## Phase Status

### 1. Data Collection — DONE
- `data.py` captures webcam frames, detects a hand each frame, crops + resizes to 128×128, accumulates into `(N, 128, 128, 3)` uint8
- Saves to `data/hands.npy` (raw) and `data/hands_augmented.npy` (~22k samples with augmentation)
- `data/flip_hands.py` — reflects `hands_augmented.npy` across the y-axis to produce `data/hands_right.npy` for right-hand VQ-VAE training
- `BBOX_PADDING = 35` used for crop context

### 2. Detection & Cropping — DONE
- `pipeline.py` — `build_detector(num_hands)` loads MediaPipe HandLandmarker; now accepts `num_hands` param (1 or 2) for dual-hand mode
- `detect_hands(frame, detector) → list[HandDetection]` — returns all detected hands (new)
- `detect_hand(frame, detector) → HandDetection` — single-hand wrapper around `detect_hands`
- `hand_types.py` — `Landmark`, `BBox`, `HandDetection` TypedDicts
- MediaPipe handedness note: in unmirrored OpenCV frames, "Right" = user's left hand, "Left" = user's right hand

### 3. VAE Model — DONE (trained)
- `model.py` — `VAE` composed of `Encoder` + `Decoder`
  - Encoder: 4× Conv2d with stride 2 (128→64→32→16→8), flatten, two linear heads for `mu` and `log_var`
  - Decoder: linear → reshape → 4× ConvTranspose2d (8→16→32→64→128), Sigmoid output
  - Latent dim: 128, ~7.7M params
- Trained on left-hand data (`hands_augmented.npy`), 100 epochs, 2× T4 (Kaggle)
- Final val loss: ~29,870 (~0.60 BCE per pixel)
- Checkpoint: `data/vae_best.pt`

**Model quality:**
- Conditioned reconstructions: accurate color, tone, position with slight blur — good for ghost effect
- Prior samples (z ~ N(0,1)): poor — latent space not fully regularized, large regions of N(0,1) unoccupied

### 4. VQ-VAE Model — DONE (working, trained on right-hand data)
- `vq_model.py` — `VQModel` composed of `VQEncoder`, `VectorQuantizer`, `VQDecoder`
  - Encoder: 3× Conv2d (128→64→32→16), channels 3→256→512→64; **no activation on final layer** (previously had ReLU which caused collapse to origin)
  - Quantizer: 512 codes × 64-dim codebook, EMA updates, cluster-split dead code revival
  - Decoder: 3× ConvTranspose2d (16→32→64→128), channels 64→512→256→3
- Checkpoint: `data/vq_best_right.pt`

**Fixes applied to resolve codebook collapse:**
1. **Removed final encoder ReLU** — ReLU restricted encoder outputs to R^64_+, creating an origin attractor where all vectors collapsed to the nearest single code. Removing it lets the encoder output unconstrained vectors
2. **Fixed DataParallel + EMA conflict** — Previously the whole VQModel was wrapped in DataParallel; EMA buffer writes on GPU 1 replicas were discarded. Now only encoder/decoder are wrapped; quantizer stays on GPU 0 and sees the full batch
3. **Fixed EMA memory leak** — EMA update lines were reassigning `self.ema_weight` and `self.ema_cluster_size` with tensors that retained the encoder's full autograd graph, chaining 150+ graphs per epoch. Wrapped all EMA + revival code in `torch.no_grad()`
4. **Cluster-split dead code revival** — Previous revival sampled from the current (collapsed) batch, reseeding all 511 dead codes to the same collapsed region. Now: find the dominant code, copy its vector into all dead slots, add N(0, 0.1²) noise per slot. Forces the dominant cluster to split

**Result:** ~79 active codes out of 512, significantly sharper reconstructions than the VAE, generalizes to face/skin patches in the padding region due to patch-level spatial bottleneck (16×16 code grid, one code per 8×8 pixel patch)

### 5. Mask Generation — DONE
- `mask.py` — two mask types:
  - `build_soft_mask()` — convex hull of 21 landmarks filled and Gaussian-blurred (feathered). Use with `-S h`
  - `build_square_mask()` — full bbox filled with 1.0 and Gaussian-blurred at edges. Use with `-S s`
- `draw_debug()` renders landmarks, bbox, and convex hull for visual verification

### 6. Blend & Display — DONE
- `main.py` — full runtime webcam loop with CLI args (see below)
- `exploration.py` — reconstruction viz, latent walk, offset preview, prior sampling, novel generation

### 7. Prior Fitting & Novel Generation — DONE
- `data/fit_prior.py` — encodes all `hands_augmented.npy` samples through the VAE encoder, computes per-dimension mean and std of the aggregate posterior means {μ_i}, saves to `data/vae_prior.npz`
- `exploration.generate_novel_images()` — loads the fitted prior, samples z ~ N(z_mean, z_std²), decodes 25 images, saves 5×5 grid to `data/vae_generation.jpg`
- This produces sensible novel hands by sampling from the occupied region of latent space rather than the full N(0,1) prior

---

## CLI Reference

### `main.py`
```
python -m VAE_vision.main [-H {l,r,lr}] [-S {h,s}] [-r [PATH]]
```
| Flag | Values | Default | Description |
|---|---|---|---|
| `-H` / `--hand` | `l`, `r`, `lr` | `l` | l=left+VAE, r=right+VQ-VAE, lr=both simultaneously |
| `-S` / `--shape` | `h`, `s` | `h` | h=convex hull mask, s=square bbox mask |
| `-r` / `--record` | optional path | `data/screen_recording` | record output to .mp4 |

### `exploration.py`
```
python -m VAE_vision.exploration [-H {l,r,lr}]
```
Runs `offset_preview` showing reconstruction pasted beside the live hand crop.

---

## File Responsibilities

| File | Status | What it owns |
|---|---|---|
| `model.py` | Done | VAE architecture — encoder, decoder, reparameterization |
| `vq_model.py` | Done | VQ-VAE — encoder (no final ReLU), EMA quantizer (cluster-split revival, no_grad EMA), decoder |
| `training.py` | Done | Training loops for VAE and VQ-VAE, HyperParams, HandDataset; DataParallel wraps encoder/decoder only for VQ |
| `data.py` | Done | Webcam image collection loop, npy save |
| `data/flip_hands.py` | Done | Horizontal flip of hands_augmented.npy → hands_right.npy |
| `data/fit_prior.py` | Done | Encode dataset through VAE, save per-dim mean/std to vae_prior.npz |
| `pipeline.py` | Done | MediaPipe detection, BGR→RGB, HandDetection dict, multi-hand support |
| `hand_types.py` | Done | TypedDicts: Landmark, BBox, HandDetection |
| `utils.py` | Done | bgr_to_rgb, rgb_to_bgr |
| `mask.py` | Done | Soft hull mask + square mask from bbox, debug overlay |
| `exploration.py` | Done | Webcam loop, offset preview (l/r/lr), reconstruction viz, latent walk, prior sampling, novel image generation |
| `main.py` | Done | Runtime webcam loop: -H hand mode, -S mask shape, -r recording |

---

## Data Files

| File | Description |
|---|---|
| `data/hands.npy` | Raw left-hand crops, ~20k samples, (N, 128, 128, 3) uint8 |
| `data/hands_augmented.npy` | Augmented left-hand crops, ~22k samples |
| `data/hands_right.npy` | Horizontally flipped hands_augmented.npy for right-hand training |
| `data/vae_best.pt` | Best VAE checkpoint (left hand) |
| `data/vq_best_right.pt` | Best VQ-VAE checkpoint (right hand) |
| `data/vae_prior.npz` | Empirical prior: per-dim mean + std of {μ_i} over training set |

---

## Tunable Knobs

- **`GHOST_ALPHA`** — decoded blend weight (0=invisible, 1=full replacement)
- **`LATENT_DIM`** (VAE) — 128 currently; lower = ghostier, higher = sharper
- **`BLUR_KERNEL_SIZE`** — GaussianBlur kernel for mask feathering (larger = softer edges)
- **`BBOX_PADDING`** — pixels of context around hand crop (currently 35)
- **`commitment_weight`** (VQ) — currently 0.25
- **`decay`** (VQ EMA) — currently 0.99

---

## Known Issues / What's Next

- **VAE prior samples poor quality** — latent space partially regularized. Use `generate_novel_images()` with `data/vae_prior.npz` for sensible outputs; raw N(0,1) sampling produces garbage in unoccupied regions
- **VQ-VAE ~79/512 codes active** — further tuning (entropy regularization, lower commitment weight) could expand codebook utilization; for the ghost pipeline this is sufficient
- **MediaPipe handedness in lr mode** — if ghost appears on the wrong hand, swap `_MEDIAPIPE_LEFT`/`_MEDIAPIPE_RIGHT` constants in `main.py`
