import cv2
import numpy as np
import torch

from VAE_vision.model import VAE
from VAE_vision.pipeline import build_detector, detect_hand
from VAE_vision.training import HyperParams

CHECKPOINT_PATH = "data/vae_epoch100.pt"
BBOX_PADDING = 35
VAE_INPUT_SIZE = 128
OFFSET_GAP = 10


def _load_model(checkpoint_path: str) -> tuple[VAE, torch.device]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.serialization.add_safe_globals([HyperParams])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = VAE(latent_dim=ckpt["hp"].latent_dim)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, device


def _reconstruct(
    crop: np.ndarray,
    model: VAE,
    device: torch.device,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    tensor = torch.from_numpy(crop).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        recon, _, _ = model(tensor)

    recon_np = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
    recon_np = (recon_np * 255).clip(0, 255).astype(np.uint8)
    return cv2.resize(recon_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def main() -> None:
    model, device = _load_model(CHECKPOINT_PATH)
    detector = build_detector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam")
        return

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"Webcam opened: {frame_w}x{frame_h}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detection = detect_hand(frame, detector)

        if detection["detected"] and detection["bbox"] is not None:
            bbox = detection["bbox"]
            x_min = max(0, bbox["x_min"] - BBOX_PADDING)
            y_min = max(0, bbox["y_min"] - BBOX_PADDING)
            x_max = min(frame_w, bbox["x_max"] + BBOX_PADDING)
            y_max = min(frame_h, bbox["y_max"] + BBOX_PADDING)

            crop = frame[y_min:y_max, x_min:x_max]
            crop_h, crop_w = crop.shape[:2]

            resized = cv2.resize(crop, (VAE_INPUT_SIZE, VAE_INPUT_SIZE), interpolation=cv2.INTER_AREA)
            recon = _reconstruct(resized, model, device, crop_h, crop_w)

            paste_x = x_max + OFFSET_GAP
            paste_x_end = paste_x + crop_w
            if paste_x_end <= frame_w:
                frame[y_min:y_max, paste_x:paste_x_end] = recon

        cv2.imshow("VAE Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
