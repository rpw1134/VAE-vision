from dataclasses import dataclass
from math import floor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

from VAE_vision.model import VAE
from VAE_vision.vq_model import VQModel


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
    val_split: float = 0.1


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
    hp: HyperParams | None = None,
) -> None:
    hp = hp or HyperParams()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    n_gpus = torch.cuda.device_count()
    print(f"Device: {device}  GPUs: {n_gpus}")

    dataset = HandDataset(npy_path)
    n_val = floor(len(dataset) * hp.val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    print(f"Dataset: {n_train} train / {n_val} val")

    loader = DataLoader(train_set, batch_size=hp.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=hp.batch_size, shuffle=False, num_workers=4, pin_memory=True)

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

    best_val_loss = float("inf")

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

        model.eval()
        val_loss = val_recon = val_kl = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                recon, mu, log_var = model(batch)
                loss, recon_loss, kl_loss = _vae_loss(recon, batch, mu, log_var, beta)
                val_loss  += loss.item()
                val_recon += recon_loss.item()
                val_kl    += kl_loss.item()

        n_val_batches = len(val_loader)
        avg_val_loss  = val_loss  / n_val_batches
        avg_val_recon = val_recon / n_val_batches
        avg_val_kl    = val_kl    / n_val_batches

        scheduler.step(avg_val_loss)

        writer.add_scalars("loss/total", {"train": avg_loss,  "val": avg_val_loss},  epoch)
        writer.add_scalars("loss/recon", {"train": avg_recon, "val": avg_val_recon}, epoch)
        writer.add_scalars("loss/kl",    {"train": avg_kl,    "val": avg_val_kl},    epoch)
        writer.add_scalar("beta", beta, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"epoch {epoch+1:03d}/{hp.epochs}  "
            f"train={avg_loss:.4f}  val={avg_val_loss:.4f}  "
            f"recon={avg_recon:.4f}  kl={avg_kl:.4f}  beta={beta:.3f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({"epoch": epoch, "model": state, "hp": hp}, ckpt_dir / "vae_best.pt")
            print(f"  -> saved best (val={best_val_loss:.4f})")

    writer.close()
    print("Training complete.")


@dataclass
class VQHyperParams:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 100
    commitment_weight: float = 0.25
    lr_factor: float = 0.5
    lr_patience: int = 5
    val_split: float = 0.1


def _vq_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    commitment_loss: torch.Tensor,
    commitment_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = F.mse_loss(recon, target, reduction="mean")
    commitment_loss = commitment_loss.mean()
    total = recon_loss + commitment_weight * commitment_loss
    return total, recon_loss, commitment_loss


def train_vq(
    npy_path: str,
    checkpoint_dir: str,
    hp: VQHyperParams | None = None,
) -> None:
    hp = hp or VQHyperParams()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    n_gpus = torch.cuda.device_count()
    print(f"Device: {device}  GPUs: {n_gpus}")

    dataset = HandDataset(npy_path)
    n_val = floor(len(dataset) * hp.val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    print(f"Dataset: {n_train} train / {n_val} val")

    loader = DataLoader(train_set, batch_size=hp.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=hp.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = VQModel()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    model = model.to(device)
    if n_gpus > 1:
        model.encoder = nn.DataParallel(model.encoder)
        model.decoder = nn.DataParallel(model.decoder)

    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=hp.lr_factor, patience=hp.lr_patience
    )

    writer = SummaryWriter()
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    first_batch = next(iter(loader)).to(device)
    with torch.no_grad():
        z = model.encoder(first_batch)
        model.quantizer.initialize_from_data(z)
    print("Codebook initialized from data.")

    for epoch in range(hp.epochs):
        model.train()
        total_loss = recon_sum = commit_sum = unique_sum = 0.0

        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            recon, commitment, unique_codes = model(batch)
            loss, recon_loss, commitment_loss = _vq_loss(
                recon, batch, commitment, hp.commitment_weight
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            recon_sum  += recon_loss.item()
            commit_sum += commitment_loss.item()
            unique_sum += unique_codes.float().mean().item()

        n_batches  = len(loader)
        avg_loss   = total_loss / n_batches
        avg_recon  = recon_sum  / n_batches
        avg_commit = commit_sum / n_batches
        avg_unique = unique_sum / n_batches

        model.eval()
        val_loss = val_recon = val_commit = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                recon, commitment, _ = model(batch)
                loss, recon_loss, commitment_loss = _vq_loss(
                    recon, batch, commitment, hp.commitment_weight
                )
                val_loss   += loss.item()
                val_recon  += recon_loss.item()
                val_commit += commitment_loss.item()

        n_val_batches  = len(val_loader)
        avg_val_loss   = val_loss   / n_val_batches
        avg_val_recon  = val_recon  / n_val_batches
        avg_val_commit = val_commit / n_val_batches

        scheduler.step(avg_val_loss)

        writer.add_scalars("loss/total",  {"train": avg_loss,   "val": avg_val_loss},   epoch)
        writer.add_scalars("loss/recon",  {"train": avg_recon,  "val": avg_val_recon},  epoch)
        writer.add_scalars("loss/commit", {"train": avg_commit, "val": avg_val_commit}, epoch)
        writer.add_scalar("codebook/unique_codes", avg_unique, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"epoch {epoch+1:03d}/{hp.epochs}  "
            f"train={avg_loss:.4f}  val={avg_val_loss:.4f}  "
            f"recon={avg_recon:.4f}  commit={avg_commit:.4f}  "
            f"unique_codes={avg_unique:.0f}/512"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            state = {k.replace(".module.", "."): v for k, v in model.state_dict().items()}
            torch.save({"epoch": epoch, "model": state, "hp": hp}, ckpt_dir / "vq_best.pt")
            print(f"  -> saved best (val={best_val_loss:.4f})")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    train(
        npy_path="data/hands.npy",
        checkpoint_dir="checkpoints",
        hp=HyperParams(),
    )
