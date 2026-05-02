import cv2
import numpy as np
import torch

from VAE_vision.pixel_cnn.model import PixelCNN
from VAE_vision.pixel_cnn.training import PixelCNNHyperParams
from VAE_vision.vq.model import VQModel
from VAE_vision.vq.training import VQHyperParams


def _load_pixelcnn(checkpoint_path: str, device: torch.device) -> PixelCNN:
    torch.serialization.add_safe_globals([PixelCNNHyperParams])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    hp = ckpt["hp"]
    model = PixelCNN(num_codes=512, embed_dim=hp.embed_dim, n_layers=hp.n_layers)
    model.load_state_dict(ckpt["model"])
    return model.eval().to(device)


def _load_vqmodel(checkpoint_path: str, device: torch.device) -> VQModel:
    torch.serialization.add_safe_globals([VQHyperParams])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = VQModel()
    model.load_state_dict(ckpt["model"])
    return model.eval().to(device)


def _sample_codes(
    model: PixelCNN,
    n: int,
    device: torch.device,
    temperature: float = 1.0,
) -> torch.Tensor:
    codes = torch.zeros(n, 16, 16, dtype=torch.long, device=device)
    with torch.no_grad():
        for i in range(16):
            for j in range(16):
                logits = model(codes)                              # (n, 512, 16, 16)
                probs = torch.softmax(logits[:, :, i, j] / temperature, dim=1)
                codes[:, i, j] = torch.multinomial(probs, 1).squeeze(1)
    return codes                                                   # (n, 16, 16)


def generate_novel_images(
    pixelcnn_checkpoint: str = "data/pixelcnn_best.pt",
    vq_checkpoint: str = "data/vq_best_right.pt",
    out_path: str = "data/vq_generation.jpg",
    n: int = 25,
    grid_cols: int = 5,
    temperature: float = 1.0,
) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    pixelcnn = _load_pixelcnn(pixelcnn_checkpoint, device)
    vq_model = _load_vqmodel(vq_checkpoint, device)

    print(f"Sampling {n} images (temperature={temperature}) ...")
    codes = _sample_codes(pixelcnn, n, device, temperature)        # (n, 16, 16)

    with torch.no_grad():
        embeddings = vq_model.quantizer.codebook(codes)            # (n, 16, 16, 64)
        embeddings = embeddings.permute(0, 3, 1, 2)                # (n, 64, 16, 16)
        decoded = vq_model.decoder(embeddings)                     # (n, 3, 128, 128)

    imgs = decoded.permute(0, 2, 3, 1).cpu().numpy()
    imgs = (imgs * 255).clip(0, 255).astype(np.uint8)

    grid_rows = (n + grid_cols - 1) // grid_cols
    pad = grid_rows * grid_cols - n
    if pad > 0:
        imgs = np.concatenate([imgs, np.zeros((pad, 128, 128, 3), dtype=np.uint8)], axis=0)

    rows = [np.hstack(imgs[r * grid_cols : (r + 1) * grid_cols]) for r in range(grid_rows)]
    grid = np.vstack(rows)

    cv2.imwrite(out_path, grid)
    print(f"Saved {n} novel images → {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PixelCNN generation")
    parser.add_argument("-n", type=int, default=25, help="number of images to generate")
    parser.add_argument("-t", "--temperature", type=float, default=1.0, help="sampling temperature")
    parser.add_argument("-o", "--out", default="data/vq_generation.jpg", help="output path")
    args = parser.parse_args()

    generate_novel_images(n=args.n, temperature=args.temperature, out_path=args.out, pixelcnn_checkpoint="data/pixelcnn_best_large.pt")
