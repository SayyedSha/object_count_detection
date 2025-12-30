import cv2
import torch
from ultralytics import YOLO


def detect_and_count_objects(
    video_path,
    output_video_path,
    model_path,
    output_size=(1280, 720),

    # ===== Label Styling =====
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.7,
    thickness=2,
    box_color=(255, 0, 0),      # Blue (BGR)
    text_color=(0, 255, 0),     # Green
):
    # -----------------------------
    # Device (GPU if available)
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO(model_path)
    model.to(device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error reading video")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_w, out_h = output_size

    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to 720p
        frame = cv2.resize(frame, (out_w, out_h))

        # -----------------------------
        # YOLO Detection
        # -----------------------------
        results = model(
            frame,
            device=device,
            conf=0.4,
            verbose=False
        )[0]

        object_count = 0

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()

            object_count = len(boxes)

            for box, cls_id in zip(boxes, classes):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[int(cls_id)]

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

                # Draw label
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    font,
                    font_scale,
                    text_color,
                    thickness,
                )

        # -----------------------------
        # Draw total count
        # -----------------------------
        cv2.putText(
            frame,
            f"Total Objects: {object_count}",
            (20, 40),
            font,
            1.0,
            (0, 0, 255),
            2
        )

        writer.write(frame)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
