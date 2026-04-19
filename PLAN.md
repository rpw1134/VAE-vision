# VAE-Vision — Project Plan

## What It Does
Applies a real-time "ghostly hand" effect to webcam video. Your hand is detected, reconstructed through a neural network, and blended back over the live feed as a translucent echo of itself.

## Phases

### 1. Model (offline)
Build and train the VAE on hand images. The network learns to compress a hand crop into a small latent vector and reconstruct it. A well-trained model produces soft, slightly-smeared reconstructions — exactly the ghost aesthetic we want.

**Packages:** `torch`, `torchvision`, `albumentations`, `Pillow`, `tensorboard`, `tqdm`

### 2. Detection & Cropping (runtime)
Use MediaPipe to locate the hand in each webcam frame. It returns 21 landmarks and a bounding box. We crop that region and feed it to the VAE.

**Packages:** `mediapipe`, `opencv-python`, `numpy`

### 3. Mask Generation (runtime)
Project the 21 landmarks into a convex hull shape and blur the edges heavily. This creates a soft mask so the ghost fades out naturally at the fingers and wrist instead of having a hard rectangular cutout.

**Packages:** `opencv-python`, `numpy`

### 4. Blend & Display (runtime)
Alpha-blend the VAE's decoded output over the original frame using the mask. Tune `GHOST_ALPHA` to control how opaque the ghost appears. Stream the result to a window at real-time frame rate.

**Packages:** `opencv-python`, `numpy`

## File Responsibilities

| File | What it owns |
|---|---|
| `model.py` | VAE architecture — encoder, decoder, reparameterization |
| `pipeline.py` | Per-frame logic — detect, crop, encode, decode, blend |
| `mask.py` | Soft mask from landmarks |
| `utils.py` | Small helpers (color conversion, bbox math) |
| `main.py` | Webcam loop, window, model loading |

## Tunable Knobs
- **`GHOST_ALPHA`** — how visible the ghost is (0 = invisible, 1 = fully replaces hand)
- **`LATENT_DIM`** — larger = more detail preserved, less ghostly
- **`BLUR_KERNEL_SIZE`** — larger = softer mask edges
- **`BBOX_PADDING`** — how much extra space around the hand crop
