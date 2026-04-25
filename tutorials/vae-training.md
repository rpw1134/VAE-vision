# VAE Training — Hyperparameters, Optimizers, and Scheduling

## The Loss Function

VAE loss has two terms that pull against each other:

```
loss = reconstruction_loss + beta * KL_loss
```

**Reconstruction loss** — measures how well the decoder reproduces the input. Binary Cross Entropy (BCE) works well when the decoder uses Sigmoid (output in [0,1]). MSE is an alternative but tends to produce blurrier reconstructions.

**KL divergence** — measures how far the learned latent distribution is from a standard normal N(0,1). It acts as a regularizer, preventing the encoder from just memorizing inputs by forcing the latent space to be smooth and continuous.

```python
kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
```

### Beta Annealing

The tension between reconstruction and KL is controlled by `beta`. A common trick is **KL annealing** — start `beta` at 0 and ramp it up over the first N epochs. This lets the network learn to reconstruct first before the KL term starts compressing the latent space.

```
epoch 0-10:  beta = 0.0   (pure reconstruction)
epoch 10-30: beta ramps 0.0 → 1.0
epoch 30+:   beta = 1.0
```

Without annealing, a high beta early on can cause **posterior collapse** — the encoder ignores the input and outputs N(0,1) for everything, making the latent space useless.

---

## Optimizer

**Adam** is the standard choice for VAEs. It adapts learning rates per-parameter and handles the noisy gradients from the reparameterization trick well.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

**AdamW** is a minor improvement — it decouples weight decay from the gradient update, which can improve generalization slightly. Worth trying if Adam overfits.

Avoid SGD for VAEs — it requires careful tuning and is generally slower to converge on this kind of task.

---

## Learning Rate

| Value | Effect |
|---|---|
| `1e-3` | Good starting point, fast convergence |
| `1e-4` | More stable, less likely to diverge |
| `1e-5` | Very conservative, use for fine-tuning |

Start at `1e-3`. If loss oscillates wildly, drop to `1e-4`.

### Learning Rate Scheduling

**ReduceLROnPlateau** — halves the LR when validation loss stops improving. Low-effort and effective:

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)
# call after each epoch:
scheduler.step(val_loss)
```

**CosineAnnealingLR** — smoothly decays LR following a cosine curve over training. Good if you know roughly how many epochs you'll train:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
```

For this project, `ReduceLROnPlateau` is the lower-effort option since training time on Kaggle T4s is bounded.

---

## Batch Size

Larger batches give more stable gradient estimates, which matters for the KL term. On a T4 with 128×128 inputs:

- **64** — safe starting point
- **128** — fits comfortably, smoother training
- **256** — possible but watch VRAM usage

Larger batch sizes also mean fewer steps per epoch, so the LR scheduler triggers less frequently — something to account for when setting `patience`.

---

## Epochs

VAEs on small datasets (a few thousand images) typically converge in **30–100 epochs**. Watch the reconstruction loss — when it plateaus and the KL term has stabilized, you're done. TensorBoard makes this easy to monitor.

A sign of good convergence: reconstructions look like soft, blurred versions of the input rather than noise or a mean-colored blob.

---

## Latent Dimension

`LATENT_DIM = 128` is a good default for 128×128 hand images. Larger values preserve more detail but reduce the ghostly smearing effect — the reconstruction will look too sharp. Smaller values (e.g. 32, 64) increase compression and produce more abstract, dream-like reconstructions.

This is the most impactful knob for controlling the aesthetic of the ghost effect.

---

## Practical Checklist for Kaggle T4s

- Normalize inputs to `[0, 1]` before passing to the model (divide by 255)
- Pin memory in the DataLoader (`pin_memory=True`) for faster GPU transfers
- Use `torch.compile(model)` if on PyTorch 2.x for a free speedup
- Save a checkpoint every N epochs so you can resume if the session times out
- Log `recon_loss` and `kl_loss` separately to TensorBoard — if KL collapses to 0, beta is too high too early

---

## Relevant Docs

- Adam optimizer: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
- ReduceLROnPlateau: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
- `torch.nn.functional.binary_cross_entropy`: https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy.html
- Beta-VAE paper (KL annealing motivation): https://openreview.net/forum?id=Sy2fchgIW
