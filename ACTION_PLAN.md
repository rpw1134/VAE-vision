# VQ-VAE Codebook Collapse — Action Plan

## TL;DR
The diagnosis in `CURRENT.md` blames the `ConvTranspose2d` decoder for being "inherently spatial." This is **partially incorrect** and is sending the search in the wrong direction. The dominant cause is much more likely the **final `ReLU` on the encoder** combined with a **dead-code revival strategy that revives into the collapsed region**. Fix those two first; everything else in the section below is fallback.

---

## Re-examining the Root Cause

`CURRENT.md` claims the decoder bypasses the bottleneck because `ConvTranspose2d` is "inherently spatial." But `ConvTranspose2d` is translation-**equivariant**: given a constant 16×16 feature map, its output is also constant in the interior, modulo a few-pixel zero-padding artifact at the borders. A single code tiled across the 16×16 grid cannot, by itself, produce a recognizable hand image in the interior. So either:

- (a) reconstructions are mostly a flat blob with edge artifacts and *appear* hand-like only because hands are vaguely centered, or
- (b) "1–4 codes" actually means 1–4 codes are spread across the 16×16 grid, giving the decoder enough crude positional information (e.g., "palm region" vs. "finger region") to render an average hand. `unique_codes` is computed across `B*H*W`, so a per-image breakdown could verify this.

Either way, the *symptom* — concentrated assignments — has identifiable causes that aren't "the decoder is too clever":

### Cause 1: Final `ReLU` on the encoder (`vq_model.py:19`)
```python
nn.Conv2d(512, 64, kernel_size=4, stride=2, padding=1),
nn.ReLU(),    # <-- restricts encoder output to R^64_+
```
Standard VQ-VAE / VQ-GAN encoders end with a **no-activation** conv (typically a 1×1 projection to the embedding dim). The ReLU here:

- Clips the entire negative half-space, so the encoder output lives in `[0, ∞)^64` — a corner of the embedding space.
- Creates a hard accumulation point at the origin: any pre-activation with all components ≤ 0 produces the zero vector. Many spatial positions land on exactly `0`, and `0` is closest to whichever codebook entry happens to be nearest the origin → deterministic assignment to one code.
- Makes "data-dependent init" useless: if early encoder outputs are mostly zero, the codebook is seeded with copies of zero.

### Cause 2: Dead-code revival reseeds *into* the collapsed region
```python
random_idx = torch.randint(0, x_flat.shape[0], (n_dead,), device=x.device)
self.codebook.weight.data[dead] = x_flat[random_idx].detach()
```
After collapse, every position in `x_flat` is a near-duplicate of the dominant encoder output. Reviving 511 dead codes from this batch reinitializes them all to vectors *near the dominant code*. Next batch they get re-assigned away → revived again → same place. This explains the observation that "codes are revived every batch but immediately lose assignments."

### Cause 3: EMA divides unused codes by ~zero, dragging them to the origin
```python
smoothed = (self.ema_cluster_size + 1e-5) / ...
self.codebook.weight.data = self.ema_weight / smoothed.unsqueeze(1)
```
For unused codes, `ema_weight → 0` and `ema_cluster_size → 0`. The ratio is numerically ~`0 / 1e-5 ≈ 0`, so unused codes drift toward the origin every step. Combined with cause 1, the origin is also where the encoder dumps everything — so the unused codes pile up exactly where they get most heavily assigned, which then makes one of them dominant. This is a positive-feedback loop.

### Cause 4: DataParallel + EMA (already noted in `CURRENT.md`)
DataParallel replicates the master module to GPU 1 each forward pass; in-place EMA buffer writes on the replica are dropped. Half the batch never informs the codebook. This isn't the *primary* collapse cause (collapse happens on a single GPU too), but it makes diagnosis noisier — fix it so you can trust the metrics.

### Cause 5: MSE rewards averaging
Pixelwise MSE has a known failure mode: when the model can't reconstruct sharply, the loss-minimizing output is the *mean* of plausible reconstructions. A collapsed codebook plus an MSE loss is a stable equilibrium — the decoder learns to produce a blurry "average hand," which under MSE is locally optimal. Perceptual loss does not have this attractor.

---

## Solutions (priority order)

Each step is independent. Run them in order, retraining for 5–10 epochs per try and watching `codebook/unique_codes`. Stop as soon as one works.

### 1. Remove the final encoder `ReLU` (largest expected effect)
**Change**: drop the trailing `nn.ReLU()` after the last `Conv2d` in `VQEncoder`. Optionally add a 1×1 projection conv for cleaner separation:
```python
nn.Conv2d(512, 64, kernel_size=1)   # final pre-quant projection, no activation
```
**Why it should work**: removes the artificial origin-attractor. Encoder outputs are now free to occupy `R^64`, codebook entries can spread to negative regions, and the symmetric init range `[-1, 1]` (which was already tried) becomes meaningful for the first time. Standard practice in every reference implementation (DeepMind, OpenAI VQ-VAE-2, Taming Transformers).

### 2. Replace dead-code revival source
Currently revives from the *current* (collapsed) batch. Two cheap alternatives:

**Option A — perturb the dominant code (cluster split)**:
```python
top_idx = self.ema_cluster_size.argmax()
self.codebook.weight.data[dead] = (
    self.codebook.weight.data[top_idx]
    + 0.1 * torch.randn(n_dead, embedding_dim, device=device)
)
```
**Option B — keep a rolling buffer of past encoder outputs**:
maintain a circular buffer (e.g. 8192 vectors) updated each step from `x_flat`; revive by sampling from the buffer. The buffer carries diversity from earlier in training before collapse.

**Why it should work**: revival has to inject *new directions* into the codebook, not duplicate the collapsed point. Option A gives the dominant cluster a chance to split (proven to work in clustering; this is essentially k-means++); option B preserves earlier-epoch variance.

### 3. Fix the DataParallel + EMA conflict
Wrap only the encoder and decoder in `DataParallel`; keep the quantizer running on GPU 0:
```python
self.encoder = nn.DataParallel(VQEncoder())
self.quantizer = VectorQuantizer()      # GPU 0 only
self.decoder = nn.DataParallel(VQDecoder())
```
In the forward pass, `gather` encoder outputs onto GPU 0 before quantizing, then `scatter` for the decoder. Or simpler: just disable DataParallel for VQ runs and use one GPU. Two T4s are not bottlenecking 5M params on 20k samples — single-GPU will run in 2–3 minutes per run.

**Why it should work**: makes EMA see the full batch and removes a confound. Won't fix collapse alone; needed for clean diagnostics.

### 4. Add Gaussian noise to encoder outputs before quantization
```python
if self.training:
    x = x + 0.1 * torch.randn_like(x)
```
**Why it should work**: hard `argmin` is a deterministic step function — once an encoder output is closer to code A than code B, it stays in A's basin even if the encoder shifts slightly. Adding noise softens this: borderline outputs flip between codes, EMA accumulates statistics for both, and codebook spread is preserved. This is a classic "explore" mechanism added to a "greedy" assignment.

### 5. Switch to cosine-distance quantization with L2-normalized vectors
Normalize encoder outputs and codebook entries to unit length, then quantize using cosine similarity (equivalent to argmax of dot product). This is the "Improved VQGAN" trick (Yu et al. 2022); it eliminates the magnitude-collapse failure mode entirely.

**Why it should work**: when encoder outputs and codebook are L2-normalized, "all collapse to origin" is geometrically impossible — the unit sphere has no origin. Code utilization in their paper went from <10% to >90% with this single change.

### 6. Switch to FSQ (Finite Scalar Quantization)
Drop the learned codebook entirely. The encoder outputs a small-dim vector (e.g. 6 dims), each dim is mapped through `tanh` then rounded to a fixed grid (e.g. 8 levels per dim → 8^6 = 262k unique codes). No codebook, no commitment loss, no EMA, no dead codes possible by construction.

**Why it should work**: collapse cannot happen — every grid point is reachable, and the quantizer has no learnable state to collapse. Mentzer et al. (2023) show FSQ matches VQ-VAE quality with a fraction of the engineering. If steps 1–5 fail, FSQ is the right exit.

### 7. Entropy regularization (already in `CURRENT.md`)
Add `-λ * H(p)` where `p` is the per-batch empirical assignment distribution (softmax over `-distances` is differentiable; one-hot is not). Useful as an *adjunct* to step 1, but on its own it just changes the equilibrium loss, not the geometry that's causing collapse — try it after the structural fixes.

---

## Suggested order of operations

1. Implement steps 1, 2A, 3 together in one commit. Re-train. Expected: code utilization climbs to dozens or hundreds within 3–5 epochs.
2. If unique codes stays under ~30, add step 4 (noise injection).
3. If still collapsing, jump to step 5 (cosine quantization). This is the highest-leverage change short of replacing the quantizer.
4. If even that fails, the problem isn't in the quantizer — it's in the encoder/decoder capacity balance. Move to step 6 (FSQ) and stop tuning VQ.

## Verification metrics to log per epoch
- `unique_codes_per_image` — `indices.view(B, -1).unique(dim=-1)` then mean. The current metric is per-batch and conflates per-image diversity with cross-image diversity.
- `code_entropy` — `H(p) = -∑ p_k log p_k` over batch usage. Goes up when codes spread.
- `encoder_output_norm` — mean L2 norm of `x_flat`. If it's collapsing toward 0, that confirms cause 1 even after the ReLU is removed.
- `dominant_code_share` — fraction of positions assigned to the most-used code. Should drop below 10% in a healthy run.

## Out of scope (do not change yet)
- Encoder channel progression `3→256→512→64` is wide-then-narrow and slightly unusual, but not load-bearing for collapse. Leave it alone until the quantizer is healthy.
- Latent spatial size 16×16. Reducing it (e.g., to 8×8) would lower information bandwidth and *increase* collapse pressure — wrong direction.
- Commitment weight 0.25. It's the canonical value; only revisit if collapse persists *after* the codebook starts spreading.
