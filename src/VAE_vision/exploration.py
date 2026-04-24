import cv2
import time

from VAE_vision.mask import draw_debug
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


if __name__ == "__main__":
    webcam_loop()
