from dataclasses import dataclass
from math import floor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

from VAE_vision.pixel_cnn.model import PixelCNN


@dataclass
class PixelCNNHyperParams:
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 64
    epochs: int = 50
    warmup_fraction: float = 0.05   # fraction of total steps used for linear warmup
    grad_clip: float = 1.0
    val_split: float = 0.1
    embed_dim: int = 128
    n_layers: int = 4


class CodeDataset(Dataset):
    def __init__(self, npy_path: str) -> None:
        raw = np.load(npy_path)                                    # (N, 16, 16) int16
        self.data = torch.from_numpy(raw.astype(np.int64))        # (N, 16, 16) int64

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def _bits_per_code(loss: float) -> float:
    """Convert mean cross-entropy (nats) to bits per code position."""
    return loss / torch.log(torch.tensor(2.0)).item()


def train_pixelcnn(
    npy_path: str = "data/vq_codes.npy",
    checkpoint_dir: str = "data",
    hp: PixelCNNHyperParams | None = None,
) -> None:
    hp = hp or PixelCNNHyperParams()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = CodeDataset(npy_path)
    n_val = floor(len(dataset) * hp.val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    print(f"Dataset: {n_train} train / {n_val} val")

    train_loader = DataLoader(train_set, batch_size=hp.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=hp.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = PixelCNN(num_codes=512, embed_dim=hp.embed_dim, n_layers=hp.n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)

    total_steps  = len(train_loader) * hp.epochs
    warmup_steps = max(1, int(total_steps * hp.warmup_fraction))
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=hp.lr * 1e-2
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps]
    )

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter()

    best_val_loss = float("inf")
    global_step   = 0

    for epoch in range(hp.epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)            # (B, 16, 16) int64
            logits = model(batch)                                   # (B, 512, 16, 16)
            loss = nn.functional.cross_entropy(logits, batch)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            global_step += 1

        avg_train = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                logits = model(batch)
                val_loss += nn.functional.cross_entropy(logits, batch).item()

        avg_val = val_loss / len(val_loader)
        current_lr = scheduler.get_last_lr()[0]

        writer.add_scalars("loss/nll",       {"train": avg_train, "val": avg_val}, epoch)
        writer.add_scalars("loss/bits",      {"train": _bits_per_code(avg_train), "val": _bits_per_code(avg_val)}, epoch)
        writer.add_scalar("lr", current_lr, epoch)

        print(
            f"epoch {epoch+1:03d}/{hp.epochs}  "
            f"train={avg_train:.4f} ({_bits_per_code(avg_train):.3f} bpc)  "
            f"val={avg_val:.4f} ({_bits_per_code(avg_val):.3f} bpc)  "
            f"lr={current_lr:.2e}"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({"epoch": epoch, "model": model.state_dict(), "hp": hp},
                       ckpt_dir / "pixelcnn_best.pt")
            print(f"  -> saved best (val={best_val_loss:.4f})")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    train_pixelcnn()
