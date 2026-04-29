import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn

from VAE_vision.hand_types import BBox, HandDetection
from VAE_vision.mask import build_soft_mask, build_square_mask
from VAE_vision.model import VAE
from VAE_vision.pipeline import build_detector, detect_hand, detect_hands
from VAE_vision.training import HyperParams, VQHyperParams
from VAE_vision.vq_model import VQModel

BBOX_PADDING = 35
VAE_INPUT_SIZE = 128
GHOST_ALPHA = 1.0

# MediaPipe labels handedness assuming a mirrored (selfie) image. OpenCV VideoCapture
# returns a non-mirrored image, so labels are swapped relative to the user's perspective:
# MediaPipe "Left"  → user's right hand
# MediaPipe "Right" → user's left hand
# If the ghost appears on the wrong hand in lr mode, swap the labels below.
_MEDIAPIPE_LEFT  = "Left"   # routes to VQ-VAE (right-hand model)
_MEDIAPIPE_RIGHT = "Right"  # routes to VAE    (left-hand  model)


def _load_vae(path: str, device: torch.device) -> VAE:
    torch.serialization.add_safe_globals([HyperParams])
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = VAE(latent_dim=ckpt["hp"].latent_dim)
    model.load_state_dict(ckpt["model"])
    return model.eval().to(device)


def _load_vqvae(path: str, device: torch.device) -> VQModel:
    torch.serialization.add_safe_globals([VQHyperParams])
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = VQModel()
    model.load_state_dict(ckpt["model"])
    return model.eval().to(device)


def _reconstruct(
    crop: np.ndarray,
    model: nn.Module,
    device: torch.device,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    tensor = torch.from_numpy(crop).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model(tensor)[0]
    recon_np = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
    recon_np = (recon_np * 255).clip(0, 255).astype(np.uint8)
    return cv2.resize(recon_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _apply_ghost(
    frame: np.ndarray,
    detection: HandDetection,
    model: nn.Module,
    device: torch.device,
    frame_h: int,
    frame_w: int,
    shape: str = "h",
) -> None:
    bbox = detection["bbox"]
    x_min = max(0, bbox["x_min"] - BBOX_PADDING)
    y_min = max(0, bbox["y_min"] - BBOX_PADDING)
    x_max = min(frame_w, bbox["x_max"] + BBOX_PADDING)
    y_max = min(frame_h, bbox["y_max"] + BBOX_PADDING)
    padded_bbox: BBox = {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}

    crop = frame[y_min:y_max, x_min:x_max]
    crop_h, crop_w = crop.shape[:2]
    if crop_h == 0 or crop_w == 0:
        return

    resized = cv2.resize(crop, (VAE_INPUT_SIZE, VAE_INPUT_SIZE), interpolation=cv2.INTER_AREA)
    recon = _reconstruct(resized, model, device, crop_h, crop_w)

    if shape == "s":
        mask = build_square_mask(padded_bbox)
    else:
        mask = build_soft_mask(detection["landmarks"], padded_bbox)

    alpha = mask * GHOST_ALPHA
    blended = alpha * recon.astype(np.float32) + (1 - alpha) * crop.astype(np.float32)
    frame[y_min:y_max, x_min:x_max] = blended.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="VAE Vision ghost hand pipeline")
    parser.add_argument(
        "-H", "--hand",
        choices=["l", "r", "lr"],
        default="l",
        help="l=left hand VAE, r=right hand VQ-VAE, lr=both simultaneously",
    )
    parser.add_argument(
        "-S", "--shape",
        choices=["h", "s"],
        default="h",
        help="mask shape: h=convex hull, s=square bbox",
    )
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    vae_model   = _load_vae("data/vae_best.pt",   device) if args.hand in ("l", "lr") else None
    vqvae_model = _load_vqvae("data/vq_best_right.pt",  device) if args.hand in ("r", "lr") else None

    detector = build_detector(num_hands=2 if args.hand == "lr" else 1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam")
        return

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"Webcam: {frame_w}x{frame_h}  mode={args.hand}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.hand == "lr":
            for detection in detect_hands(frame, detector):
                hand = detection["handedness"]
                if hand == _MEDIAPIPE_RIGHT and vae_model is not None:
                    _apply_ghost(frame, detection, vae_model, device, frame_h, frame_w, args.shape)
                elif hand == _MEDIAPIPE_LEFT and vqvae_model is not None:
                    _apply_ghost(frame, detection, vqvae_model, device, frame_h, frame_w, args.shape)
        else:
            detection = detect_hand(frame, detector)
            expected = _MEDIAPIPE_RIGHT if args.hand == "l" else _MEDIAPIPE_LEFT
            if detection["detected"] and detection["bbox"] is not None and detection["handedness"] == expected:
                model = vae_model if args.hand == "l" else vqvae_model
                _apply_ghost(frame, detection, model, device, frame_h, frame_w, args.shape)

        cv2.imshow("VAE Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
