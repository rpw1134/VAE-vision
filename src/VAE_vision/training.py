from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from VAE_vision.model import VAE


@dataclass
class HyperParams:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 40
    latent_dim: int = 128
    beta_start: float = 0.0
    beta_end: float = 1.0
    beta_warmup_epochs: int = 20
    lr_factor: float = 0.5
    lr_patience: int = 5


class HandDataset(Dataset):
    def __init__(self, npy_path: str) -> None:
        raw = np.load(npy_path)                              # (N, H, W, 3) uint8
        tensor = torch.from_numpy(raw).float() / 255.0      # (N, H, W, 3) float [0,1]
        self.data = tensor.permute(0, 3, 1, 2)              # (N, 3, H, W)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def _beta_for_epoch(epoch: int, hp: HyperParams) -> float:
    if epoch >= hp.beta_warmup_epochs:
        return hp.beta_end
    return hp.beta_start + (hp.beta_end - hp.beta_start) * (epoch / hp.beta_warmup_epochs)


def _vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = F.binary_cross_entropy(recon, target, reduction="sum") / target.size(0)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / target.size(0)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train(
    npy_path: str,
    checkpoint_dir: str,
    hp: HyperParams = field(default_factory=HyperParams),
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    n_gpus = torch.cuda.device_count()
    print(f"Device: {device}  GPUs: {n_gpus}")

    dataset = HandDataset(npy_path)
    loader = DataLoader(
        dataset,
        batch_size=hp.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model: nn.Module = VAE(latent_dim=hp.latent_dim)
    if n_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=hp.lr_factor, patience=hp.lr_patience
    )

    writer = SummaryWriter()
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(hp.epochs):
        beta = _beta_for_epoch(epoch, hp)
        model.train()
        total_loss = recon_sum = kl_sum = 0.0

        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            recon, mu, log_var = model(batch)
            loss, recon_loss, kl_loss = _vae_loss(recon, batch, mu, log_var, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            recon_sum += recon_loss.item()
            kl_sum += kl_loss.item()

        n_batches = len(loader)
        avg_loss  = total_loss / n_batches
        avg_recon = recon_sum  / n_batches
        avg_kl    = kl_sum     / n_batches

        scheduler.step(avg_loss)

        writer.add_scalar("loss/total", avg_loss,  epoch)
        writer.add_scalar("loss/recon", avg_recon, epoch)
        writer.add_scalar("loss/kl",    avg_kl,    epoch)
        writer.add_scalar("beta",       beta,      epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"epoch {epoch+1:03d}/{hp.epochs}  "
            f"loss={avg_loss:.4f}  recon={avg_recon:.4f}  kl={avg_kl:.4f}  beta={beta:.3f}"
        )

        state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save({"epoch": epoch, "model": state, "hp": hp}, ckpt_dir / f"vae_epoch{epoch+1:03d}.pt")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    train(
        npy_path="data/hands.npy",
        checkpoint_dir="checkpoints",
        hp=HyperParams(),
    )
