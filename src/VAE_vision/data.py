import cv2
import numpy as np
import albumentations as A

from VAE_vision.pipeline import build_detector, detect_hand


def _build_augmentation_pipeline() -> A.Compose:
    return A.Compose([
        A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT, p=0.8),
        A.Affine(scale=(0.7, 1.3), translate_percent=0.1, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=40, p=0.8),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(8, 24), hole_width_range=(8, 24), p=0.3),
    ])

VAE_INPUT_SIZE = 128
BBOX_PADDING = 35


def collect_images(n_samples: int = 2000, save_path: str = "data/hands.npy") -> None:
    detector = build_detector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam")
        return

    buffer: list[np.ndarray] = []
    print(f"Collecting {n_samples} samples. Press 'q' to quit early.")

    while len(buffer) < n_samples:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: failed to read frame")
            break

        detection = detect_hand(frame, detector)
        if not detection["detected"] or detection["bbox"] is None:
            continue

        bbox = detection["bbox"]
        h, w = frame.shape[:2]
        x_min = max(0, bbox["x_min"] - BBOX_PADDING)
        y_min = max(0, bbox["y_min"] - BBOX_PADDING)
        x_max = min(w, bbox["x_max"] + BBOX_PADDING)
        y_max = min(h, bbox["y_max"] + BBOX_PADDING)

        crop = frame[y_min:y_max, x_min:x_max]
        resized = cv2.resize(crop, (VAE_INPUT_SIZE, VAE_INPUT_SIZE), interpolation=cv2.INTER_AREA)
        buffer.append(resized)

        if len(buffer) % 100 == 0:
            print(f"Collected {len(buffer)}/{n_samples}")

        cv2.imshow("collecting", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    dataset = np.stack(buffer, axis=0)  # (N, 128, 128, 3)
    np.save(save_path, dataset)
    print(f"Saved {dataset.shape} array to {save_path}")


def augment_dataset(
    input_path: str = "data/hands.npy",
    output_path: str = "data/hands_augmented.npy",
    augmentations_per_image: int = 10,
) -> None:
    raw = np.load(input_path)
    n = len(raw)
    print(f"Loaded {n} images from {input_path}. Generating {n * augmentations_per_image} augmented samples...")

    pipeline = _build_augmentation_pipeline()
    buffer: list[np.ndarray] = []

    for i, image in enumerate(raw):
        for _ in range(augmentations_per_image):
            result = pipeline(image=image)
            buffer.append(result["image"])

        if (i + 1) % 100 == 0:
            print(f"Augmented {i + 1}/{n} images")

    augmented = np.stack(buffer, axis=0)
    combined = np.concatenate([raw, augmented], axis=0)
    np.save(output_path, combined)
    print(f"Saved {combined.shape} array to {output_path}  ({n} original + {len(augmented)} augmented)")


def visualize_hand_from_npy(path: str, index: int) -> None:
    dataset = np.load(path)
    frame = dataset[index]
    cv2.imshow(f"{path}[{index}]", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # collect_images()
    # visualize_hand_from_npy("data/hands.npy", 64)
    augment_dataset()
