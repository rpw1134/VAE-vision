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

## Phase 8: PixelCNN Prior for VQ-VAE Generation — Plan

The VQ-VAE has no generative prior: random sampling from the codebook ignores the learned spatial structure and produces incoherent outputs. A PixelCNN learns an autoregressive prior P(z₁, z₂, …, z₂₅₆) over the 16×16 grid of code indices, enabling novel image generation.

### Step 1 — Encode dataset to code index grids

Run every image in `data/hands_right.npy` (the right-hand training set) through the frozen VQ-VAE encoder + quantizer. Record only the `indices` tensor returned by the quantizer, not the quantized embeddings. Save to `data/vq_codes.npy` as `(N, 16, 16)` int16.

- Load `vq_best_right.pt`, set `eval()`, `torch.no_grad()`
- Batch through the full dataset (the quantizer is already in VQModel.forward — extract indices by running encoder + distances argmin directly, or add a `encode_to_indices` method to VQModel)
- This is a one-time offline step; takes seconds on MPS

### Step 2 — PixelCNN architecture (`pixelcnn.py`)

A PixelCNN operates on the 16×16 integer code grid. The core idea: masked convolutions enforce that predicting position (i, j) can only attend to positions before it in raster order.

**Two mask types:**
- **Type A** (first layer only): masks out the current pixel — can only see strictly prior pixels
- **Type B** (all subsequent layers): includes the current pixel — can see prior pixels + current layer's own activation

**Architecture:**
```
EmbedLayer:  code index → (B, D, 16, 16)    D=256 embedding dim
MaskedConv2d (Type A, D→D)                   + LayerNorm + ReLU  ×1
MaskedConv2d (Type B, D→D)                   + LayerNorm + ReLU  ×N  (N≈6–8)
1×1 Conv (D→512)                             logits over codebook
```
- Kernel size 3×3 for masked convs, 7×7 for the first layer is also common (larger receptive field at minimal cost)
- Residual connections on Type B layers
- Output: `(B, 512, 16, 16)` — 512-way softmax at every position

**Masked Conv2d implementation:**
Register a mask buffer in `__init__`, multiply `self.weight * self.mask` in `forward`. The mask is all-ones for rows above center, half-ones for the center row (Type A: up to but excluding center column; Type B: up to and including), zeros below.

### Step 3 — Training (`pixelcnn_train.py` or add to `training.py`)

- **Dataset:** load `data/vq_codes.npy`, return each `(16, 16)` int grid; target is the same grid shifted — i.e., input is the code at position (i,j) and target is the code to be predicted at (i,j)
- **Loss:** cross-entropy between `(B, 512, H, W)` logits and `(B, H, W)` target indices
- **Optimizer:** Adam, lr=1e-3, batch size 64, ~50–100 epochs on ~22k samples
- **Checkpoint:** save to `data/pixelcnn.pt`
- Log per-epoch NLL (nats or bits-per-code) — should converge to well below `log(512) ≈ 6.2 nats` if the prior has structure

### Step 4 — Autoregressive sampling

```python
def sample_pixelcnn(model, n, device):
    codes = torch.zeros(n, 16, 16, dtype=torch.long, device=device)
    for i in range(16):
        for j in range(16):
            with torch.no_grad():
                logits = model(codes)              # (B, 512, 16, 16)
            probs = torch.softmax(logits[:, :, i, j], dim=1)
            codes[:, i, j] = torch.multinomial(probs, 1).squeeze(1)
    return codes                                   # (B, 16, 16)
```

Then: look up quantized embeddings for each code, reshape to `(B, 64, 16, 16)`, pass through `VQDecoder` → `(B, 3, 128, 128)` novel image.

Temperature scaling: `logits / temperature` before softmax. `T < 1` → sharper/less diverse; `T > 1` → more random. Start at T=1.0.

### Step 5 — Integration into `exploration.py`

Add `generate_vq_novel_images(prior_path, vq_checkpoint, pixelcnn_checkpoint, n, out_path)` mirroring the existing `generate_novel_images()` for the VAE:
1. Load VQ-VAE decoder + codebook
2. Load PixelCNN
3. Sample code grids autoregressively
4. Decode to images
5. Save 5×5 grid to `data/vq_generation.jpg`

Add `-G` flag to `exploration.py` argparse to trigger this.

### Complexity / time estimates

| Step | Effort |
|---|---|
| Encode dataset | ~30 min (code + run) |
| PixelCNN architecture | ~2 hrs |
| Training script | ~1 hr |
| Sampling + integration | ~1 hr |

### Key risk: only ~79/512 codes active

With 79 active codes the PixelCNN is modeling a distribution over a ~79-symbol alphabet embedded in a 512-size space. The model will learn this naturally (the inactive codes will get near-zero probability mass), but generation quality is bounded by the VQ-VAE reconstruction quality. Expanding codebook utilization before training the PixelCNN would improve results — but the prior will work regardless.

---

## Known Issues / What's Next

- **VAE prior samples poor quality** — latent space partially regularized. Use `generate_novel_images()` with `data/vae_prior.npz` for sensible outputs; raw N(0,1) sampling produces garbage in unoccupied regions
- **VQ-VAE ~79/512 codes active** — further tuning (entropy regularization, lower commitment weight) could expand codebook utilization; for the ghost pipeline this is sufficient
- **MediaPipe handedness in lr mode** — if ghost appears on the wrong hand, swap `_MEDIAPIPE_LEFT`/`_MEDIAPIPE_RIGHT` constants in `main.py`
- **VQ-VAE generation** — PixelCNN prior needed; plan outlined in Phase 8 above
