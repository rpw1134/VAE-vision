"""
Encode the right-hand image dataset to VQ code index grids.
Runs hands_right.npy through the frozen VQ-VAE encoder + quantizer and saves
the resulting (N, 16, 16) int16 index array to data/vq_codes.npy.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from VAE_vision.data import HandDataset
from VAE_vision.vq.model import VQModel
from VAE_vision.vq.training import VQHyperParams


def encode_dataset(
    npy_path: str = "data/hands_right.npy",
    checkpoint_path: str = "data/vq_best_right.pt",
    out_path: str = "data/vq_codes.npy",
    batch_size: int = 256,
) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    torch.serialization.add_safe_globals([VQHyperParams])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = VQModel()
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    dataset = HandDataset(npy_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"Encoding {len(dataset)} images from {npy_path} ...")

    all_indices: list[np.ndarray] = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            indices = model.encode_to_indices(batch)   # (B, 16, 16)
            all_indices.append(indices.cpu().numpy().astype(np.int16))
            if (i + 1) % 10 == 0:
                print(f"  {(i + 1) * batch_size}/{len(dataset)}")

    codes = np.concatenate(all_indices, axis=0)        # (N, 16, 16)
    np.save(out_path, codes)
    print(f"Saved {codes.shape} int16 array → {out_path}")


if __name__ == "__main__":
    encode_dataset()
