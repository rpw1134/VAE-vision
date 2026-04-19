# VAE-Vision — Agent Reference

## Project Goal
Real-time "ghostly hand" effect: capture webcam frames, detect the hand with MediaPipe, pass the cropped hand through a Variational Autoencoder, and alpha-blend the decoded reconstruction back over the original frame using a soft landmark mask.

## Package Layout
```
src/vae_vision/
  main.py          # entry point — webcam loop
  model.py         # VAE architecture (encoder, reparameterize, decoder)
  pipeline.py      # per-frame orchestration (detect → crop → encode → decode → blend)
  mask.py          # soft mask construction from landmarks (convex hull + Gaussian blur)
  utils.py         # BGR↔RGB helpers, bbox utilities, resize with padding
```

## Per-Frame Pipeline (pipeline.py)

1. **BGR → RGB** — OpenCV captures BGR; MediaPipe requires RGB. Convert with `cv2.cvtColor`.
2. **HandLandmarker** — Run `mediapipe.tasks.vision.HandLandmarker` on the RGB frame. If no hand detected, yield the original frame unchanged and continue.
3. **Crop & resize** — Compute a bounding box from landmark pixel coords (with padding), crop from the original frame, resize to `VAE_INPUT_SIZE` (e.g. 128×128).
4. **Encode** — Normalize crop to `[0, 1]`, pass through encoder to get `(mu, log_var)`.
5. **Reparameterize & walk latent** — Sample `z = mu + eps * exp(0.5 * log_var)`. Optionally perturb `z` slightly for a ghostly drift effect.
6. **Decode** — Pass `z` through decoder, denormalize to `[0, 255]` uint8.
7. **Build soft mask (mask.py)** — Project 21 landmarks to pixel coords within the bbox, compute convex hull via `cv2.convexHull`, fill on a blank canvas, apply `cv2.GaussianBlur` to feather edges.
8. **Resize decoded output** — Scale decoded image back to original bbox dimensions.
9. **Alpha-blend** — `out = mask * decoded + (1 - mask) * original` for each pixel in the bbox region. Mask is float `[0, 1]`.
10. **Display** — Write blended region back into the frame, show with `cv2.imshow`.

## VAE Architecture (model.py)

- **Input:** `(B, 3, H, W)` float32 tensor, values in `[0, 1]`.
- **Encoder:** Conv2d stack → flatten → two linear heads (`mu`, `log_var`). Latent dim configurable (default 128).
- **Decoder:** Linear → reshape → ConvTranspose2d stack → Sigmoid output.
- **Loss (training):** `BCE_reconstruction + beta * KL_divergence`. Beta defaults to 1.0; can be annealed.
- **Reparameterization:** `z = mu + torch.randn_like(std) * std` where `std = exp(0.5 * log_var)`.

## Key Configuration Constants
```python
VAE_INPUT_SIZE   = 128          # spatial dim fed to VAE
LATENT_DIM       = 128          # z vector length
BBOX_PADDING     = 20           # px padding around hand bbox
GHOST_ALPHA      = 0.6          # decoded blend weight (0=invisible, 1=full replace)
BLUR_KERNEL_SIZE = 51           # GaussianBlur kernel for mask feathering
DEVICE           = "mps"        # or "cuda" / "cpu"
```

## Dependencies & Their Roles
| Package | Role |
|---|---|
| `torch` / `torchvision` | VAE model definition, tensor ops, training loop |
| `mediapipe` | Hand landmark detection (21 keypoints + bbox) |
| `opencv-python` | Webcam capture, BGR↔RGB, drawing, display |
| `numpy` | Array manipulation bridging OpenCV ↔ PyTorch |
| `Pillow` | Image I/O for training data loading |
| `albumentations` | Augmentation pipeline for training hand images |
| `tensorboard` | Training loss visualization |
| `tqdm` | Training epoch/batch progress bars |
| `matplotlib` | Offline visualization / debugging latent space |

## Data Flow Types
- OpenCV frames: `np.ndarray (H, W, 3)` uint8 BGR
- MediaPipe input: `np.ndarray (H, W, 3)` uint8 RGB
- VAE input tensor: `torch.Tensor (1, 3, H, W)` float32 `[0,1]`
- Mask: `np.ndarray (H, W, 1)` float32 `[0,1]`

## Training Notes
- Train the VAE offline on a hand image dataset (e.g. HaGRID, custom captures).
- Save checkpoint as `checkpoints/vae.pt` — `pipeline.py` loads from this path at startup.
- Albumentations handles flips, color jitter, and affine transforms to improve generalization.

## Coding Conventions
- All model code in `model.py`; no model logic in `pipeline.py`.
- Frame processing functions are pure (frame in → frame out); side effects only in `main.py`.
- Type-annotate all public function signatures.
- No global mutable state outside `main.py`.
