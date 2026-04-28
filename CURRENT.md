# VAE-Vision — Current State

## What It Does
Applies a real-time "ghostly hand" effect to webcam video. The hand is detected, its crop is encoded and decoded through a VAE, and the reconstruction is alpha-blended back over the live feed as a translucent echo of itself.

---

## Phase Status

### 1. Data Collection — DONE
- `data.py` captures webcam frames, detects a hand each frame, crops + resizes to 128×128, and accumulates into a NumPy array `(N, 128, 128, 3)` uint8
- Saves to `data/hands.npy` as a single file
- `BBOX_PADDING = 35` used for crop context
- ~20k augmented samples collected

### 2. Detection & Cropping — DONE
- `pipeline.py` — `build_detector()` loads MediaPipe HandLandmarker from `hand_landmarker.task`
- `detect_hand(frame, detector)` converts BGR→RGB, runs detection, returns a typed `HandDetection` dict with `detected`, `landmarks` (21 pixel-coord points), `bbox`, and `handedness`
- `utils.py` — `bgr_to_rgb` / `rgb_to_bgr` helpers
- `hand_types.py` — `Landmark`, `BBox`, `HandDetection` TypedDicts for linting

### 3. VAE Model — DONE (trained)
- `model.py` — `VAE` composed of `Encoder` + `Decoder`
  - Encoder: 4× Conv2d with stride 2 (128→64→32→16→8), flatten, two linear heads for `mu` and `log_var`
  - Decoder: linear → reshape → 4× ConvTranspose2d (8→16→32→64→128), Sigmoid output
  - Latent dim: 128, ~7.7M params
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

### 4. Mask Generation — DONE
- `mask.py` — `build_soft_mask()` fills convex hull of 21 landmarks on a blank canvas, applies `cv2.GaussianBlur` to feather edges, returns float32 `(H, W, 1)` mask in `[0, 1]`
- `draw_debug()` renders landmarks, bbox, and convex hull for visual verification

### 5. Blend & Display — DONE
- `main.py` — full runtime webcam loop: loads checkpoint, detects hand, crops, reconstructs via VAE, builds soft mask, alpha-blends decoded output over original frame, displays with `cv2.imshow`

### 6. VQ-VAE — IN PROGRESS (codebook collapse)
- `vq_model.py` — `VQModel` composed of `VQEncoder`, `VectorQuantizer`, `VQDecoder`
  - Encoder: 3× Conv2d (128→64→32→16), channels 3→256→512→64, ~5.3M params
  - Quantizer: 512 codes × 64-dim codebook, EMA updates, dead code revival
  - Decoder: 3× ConvTranspose2d (16→32→64→128), channels 64→512→256→3
- `train_vq()` in `training.py` — MSE reconstruction loss + β×commitment loss, logs unique code usage per epoch

---

## VQ-VAE: What We've Tried & Diagnostics

### The failure: codebook collapse to 1–4 codes
Every training run collapses to 1–4 active codes out of 512 within the first 2 epochs. The decoder reconstructs a reasonable average hand image without meaningfully using code content.

### Root cause
The `ConvTranspose2d` decoder is inherently spatial — it upsamples from a 16×16 grid and can learn position-dependent features independent of code content. Once the decoder learns to reconstruct from implicit spatial position alone, the encoder has no pressure to diversify its codes. This is posterior collapse through the decoder bypassing the bottleneck.

This is confirmed by `commit → 0`: the encoder collapses its outputs to a single point in space, perfectly matching one code, and reconstruction loss keeps improving anyway.

### What we tried

**1. Gradient-based codebook updates (original)**
Loss = `recon + codebook_loss + β×commitment_loss`. Produced explosive instability — commitment and codebook losses growing to millions within 5 epochs. Encoder and codebook race each other; AdamW momentum causes overshooting.

**2. L2 normalization of encoder outputs**
Constrained encoder outputs to the unit hypersphere. Slightly stabilized early training but didn't prevent collapse and changes the geometry of learned representations — not what the paper does.

**3. Codebook init uniform `[-1, 1]`**
Default `nn.Embedding` is N(0,1). ReLU encoder outputs are non-negative, so half the codebook was unreachable. Switching to `[-1, 1]` helped slightly but wasn't sufficient.

**4. BCE → MSE reconstruction loss**
BCE with sum reduction produced ~30K per-batch loss values, completely dominating the other terms. MSE with mean reduction brings all three loss terms to the same scale (~0.08). Eliminated the training instability.

**5. EMA codebook updates**
Replaced gradient-based codebook updates with exponential moving average (decay=0.99). Eliminates the encoder/codebook race — codebook smoothly tracks where encoder outputs land rather than chasing them with momentum. Stabilized training completely (losses no longer explode). Did not fix collapse.

**6. Codebook init uniform `[0, 1]`**
Matched init range to ReLU encoder output range. Marginal improvement.

**7. Dead code revival**
After each EMA update, any code with `ema_cluster_size < 1.0` gets reinitialized to a random encoder output from the current batch. Codes are revived every batch but immediately lose assignments again — the encoder has been trained toward the dominant code and all outputs cluster there.

**8. Data-dependent initialization**
Before epoch 1, run a batch through the encoder and seed all 512 codes from actual encoder outputs. Got 4 unique codes in epoch 1, collapsed to 1 by epoch 2. Confirms the encoder shifts rapidly during training to exploit the decoder's spatial bypass.

### Known remaining issues

**DataParallel + EMA conflict**
DataParallel replicates the master module (GPU 0) to GPU 1 before each forward pass. EMA buffer updates on GPU 1's replica are discarded — they don't propagate back. Only GPU 0 accumulates EMA statistics, and only from half the batch. Codebook updates are therefore based on half the data.

### What's next
Two approaches to try, in order:

1. **Fix DataParallel EMA** — run the quantizer only on GPU 0 (wrap encoder/decoder in DataParallel but keep quantizer on device 0). Full batch informs EMA.
2. **Entropy regularization** — add `-weight × H(p)` to the loss where `H(p)` is the entropy of per-batch code usage. Directly penalizes concentrated code usage, forcing the encoder to spread assignments.

---

## File Responsibilities

| File | Status | What it owns |
|---|---|---|
| `model.py` | Done | VAE architecture — encoder, decoder, reparameterization |
| `vq_model.py` | In progress | VQ-VAE architecture — encoder, vector quantizer (EMA), decoder |
| `training.py` | Done | Training loops for both VAE and VQ-VAE, HyperParams, HandDataset |
| `data.py` | Done | Image collection loop, npy save, visualizer |
| `pipeline.py` | Done | MediaPipe detection, BGR→RGB, HandDetection dict |
| `hand_types.py` | Done | TypedDicts: Landmark, BBox, HandDetection |
| `utils.py` | Done | bgr_to_rgb, rgb_to_bgr |
| `mask.py` | Done | Soft mask from landmarks, debug overlay |
| `exploration.py` | Done | Webcam loop, reconstruction viz, latent walk, prior sampling; supports both VAE and VQModel via model_cls param |
| `main.py` | Done | Runtime webcam loop with full ghost pipeline |

---

## Tunable Knobs

- **`GHOST_ALPHA`** — decoded blend weight (0 = invisible, 1 = full replacement)
- **`LATENT_DIM`** — 128 currently; lower = ghostier, higher = sharper
- **`BLUR_KERNEL_SIZE`** — GaussianBlur kernel for mask feathering (larger = softer edges)
- **`BBOX_PADDING`** — pixels of context around hand crop (currently 35)
- **`beta_end`** — raise to 2.0–4.0 + shorten warmup to tighten prior if more unconditioned quality is needed
- **`commitment_weight`** (VQ) — currently 0.25; lower = less pressure on encoder to stay near codebook
- **`decay`** (VQ) — EMA decay, currently 0.99; lower = codebook tracks encoder faster but noisier
