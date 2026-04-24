import cv2
import numpy as np

from VAE_vision.pipeline import build_detector, detect_hand

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


def visualize_hand_from_npy(path: str, index: int) -> None:
    dataset = np.load(path)
    frame = dataset[index]
    cv2.imshow(f"{path}[{index}]", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    collect_images()
    visualize_hand_from_npy("data/hands.npy", 64)
