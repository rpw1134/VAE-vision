import cv2
import numpy as np
import time
import torch

from VAE_vision.mask import draw_debug
from VAE_vision.model import VAE
from VAE_vision.pipeline import build_detector, detect_hand


def webcam_loop() -> None:
    detector = build_detector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam")
        return

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"Webcam opened: {w}x{h}")

    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: failed to read frame")
            break

        frame_count += 1
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now

        detection = detect_hand(frame, detector)
        draw_debug(frame, detection)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("explore", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Total frames: {frame_count}")


def visualize_reconstructions(
    npy_path: str = "data/hands.npy",
    checkpoint_path: str = "data/vae_best.pt",
    indices: list[int] | None = None,
    n: int = 8,
) -> None:
    from VAE_vision.training import HyperParams

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.serialization.add_safe_globals([HyperParams])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = VAE(latent_dim=ckpt["hp"].latent_dim)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    dataset = np.load(npy_path)
    if indices is None:
        indices = np.random.choice(len(dataset), size=n, replace=False).tolist()

    raws = dataset[indices]                                           # (n, 128, 128, 3)
    tensor = torch.from_numpy(raws).float() / 255.0                  # (n, 128, 128, 3)
    tensor = tensor.permute(0, 3, 1, 2).to(device)                  # (n, 3, 128, 128)

    with torch.no_grad():
        recons, _, _ = model(tensor)

    recons_np = recons.permute(0, 2, 3, 1).cpu().numpy()
    recons_np = (recons_np * 255).clip(0, 255).astype(np.uint8)

    rows = [np.hstack([raws[i], recons_np[i]]) for i in range(len(indices))]
    grid = np.vstack(rows)

    cv2.imshow("original | reconstruction", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_latent_variance(
    npy_path: str = "data/hands.npy",
    checkpoint_path: str = "data/vae_best.pt",
    index: int = 10,
    n_samples: int = 8,
) -> None:
    from VAE_vision.training import HyperParams

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.serialization.add_safe_globals([HyperParams])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = VAE(latent_dim=ckpt["hp"].latent_dim)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    raw = np.load(npy_path)[index]                                    # (128, 128, 3) uint8
    tensor = torch.from_numpy(raw).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)         # (1, 3, 128, 128)

    with torch.no_grad():
        mu, log_var = model.encoder(tensor)
        samples = [model.decoder(model.reparameterize(mu, log_var)) for _ in range(n_samples)]

    recons = [s.squeeze(0).permute(1, 2, 0).cpu().numpy() for s in samples]
    recons = [(r * 255).clip(0, 255).astype(np.uint8) for r in recons]

    grid = np.hstack([raw] + recons)
    cv2.imshow(f"original | {n_samples}x samples from same mu/log_var", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def latent_space_walk(
    npy_path: str = "data/hands.npy",
    checkpoint_path: str = "data/vae_best.pt",
    index: int = 10,
    step_size: float = 0.3,
    n_steps: int = 20,
) -> None:
    """
    Random walk through latent space starting from the encoded mu of a conditioning image.
    Each frame adds a small Gaussian perturbation to z and decodes it.
    Press 'r' to reset to mu, 'q' to quit.
    """
    from VAE_vision.training import HyperParams

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.serialization.add_safe_globals([HyperParams])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = VAE(latent_dim=ckpt["hp"].latent_dim)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    raw = np.load(npy_path)[index]
    tensor = torch.from_numpy(raw).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        mu, _ = model.encoder(tensor)

    z = mu.clone()
    frames = []

    with torch.no_grad():
        for _ in range(n_steps):
            z = z + step_size * torch.randn_like(z)
            decoded = model.decoder(z).squeeze(0).permute(1, 2, 0).cpu().numpy()
            frames.append((decoded * 255).clip(0, 255).astype(np.uint8))

    grid = np.hstack([raw] + frames)
    cv2.imshow(f"latent walk — {n_steps} steps from mu", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_prior_samples(
    checkpoint_path: str = "data/vae_best.pt",
    n: int = 16,
) -> None:
    from VAE_vision.training import HyperParams

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.serialization.add_safe_globals([HyperParams])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = VAE(latent_dim=ckpt["hp"].latent_dim)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    with torch.no_grad():
        z = torch.randn(n, ckpt["hp"].latent_dim, device=device)
        decoded = model.decoder(z)

    frames = decoded.permute(0, 2, 3, 1).cpu().numpy()
    frames = (frames * 255).clip(0, 255).astype(np.uint8)

    grid = np.hstack(frames)
    cv2.imshow(f"{n} samples from prior N(0,1)", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def offset_preview(
    checkpoint_path: str = "data/vae_best.pt",
    bbox_padding: int = 35,
    offset_gap: int = 10,
) -> None:
    from VAE_vision.training import HyperParams

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.serialization.add_safe_globals([HyperParams])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = VAE(latent_dim=ckpt["hp"].latent_dim)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    detector = build_detector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam")
        return

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detection = detect_hand(frame, detector)
        if detection["detected"] and detection["bbox"] is not None:
            bbox = detection["bbox"]
            x_min = max(0, bbox["x_min"] - bbox_padding)
            y_min = max(0, bbox["y_min"] - bbox_padding)
            x_max = min(frame_w, bbox["x_max"] + bbox_padding)
            y_max = min(frame_h, bbox["y_max"] + bbox_padding)

            crop = frame[y_min:y_max, x_min:x_max]
            crop_h, crop_w = crop.shape[:2]

            resized = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
            tensor = torch.from_numpy(resized).float() / 255.0
            tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                recon, _, _ = model(tensor)
            recon_np = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
            recon_np = (recon_np * 255).clip(0, 255).astype(np.uint8)
            recon_np = cv2.resize(recon_np, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

            paste_x = x_max + offset_gap
            paste_x_end = paste_x + crop_w
            if paste_x_end <= frame_w:
                frame[y_min:y_max, paste_x:paste_x_end] = recon_np

        cv2.imshow("offset preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    offset_preview()
