import cv2
import torch
from collections import defaultdict
from ultralytics import YOLO
import app.cloudinary_uploader as cd

allowed_classes = ["person", "car", "truck", "motorcycle", "boat"]

def resize_with_aspect_ratio(image, target_size):
    target_w, target_h = target_size
    h, w = image.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    pad_w = target_w - new_w
    pad_h = target_h - new_h

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    return cv2.copyMakeBorder(
        resized,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )


def detect_and_count_objects(
    video_path,
    output_video_path,
    model_path,
    output_size=(1280, 720),

    # Styling
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.7,
    thickness=2,
    box_color=(255, 0, 0),
    text_color=(0, 255, 0),
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO(model_path)
    model.to(device)

    # -----------------------------
    # Convert allowed class names → IDs
    # -----------------------------
    if allowed_classes:
        allowed_class_ids = {
            k for k, v in model.names.items() if v in allowed_classes
        }
    else:
        allowed_class_ids = None  # allow all

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

    class_id_map = defaultdict(set)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = resize_with_aspect_ratio(frame, (out_w, out_h))

        results = model.track(
            frame,
            device=device,
            persist=True,
            conf=0.4,
            verbose=False
        )[0]

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy()

            for box, cls_id, track_id in zip(boxes, classes, track_ids):
                cls_id = int(cls_id)

                # ✅ Filter classes
                if allowed_class_ids is not None and cls_id not in allowed_class_ids:
                    continue

                class_name = model.names[cls_id]
                class_id_map[class_name].add(int(track_id))

                x1, y1, x2, y2 = map(int, box)

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
                cv2.putText(
                    frame,
                    f"{class_name} #{int(track_id)}",
                    (x1, y1 - 10),
                    font,
                    font_scale,
                    text_color,
                    thickness,
                )

        # Live counts
        y_offset = 40
        for cls, ids in class_id_map.items():
            cv2.putText(
                frame,
                f"{cls}: {len(ids)}",
                (20, y_offset),
                font,
                0.8,
                (0, 0, 255),
                2
            )
            y_offset += 30

        writer.write(frame)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    return {cls: len(ids) for cls, ids in class_id_map.items()}

processed_video = None
def getvideo(video):
    # print(video)
    if video == "demo-1.mp4":
        processed_video = cd.get_cloudinary_playback_url("processed_videos/puvgat92h1qdqphgucrz")

        return {
            "status": "success",
            "video_url": processed_video,
            "counts": [
                {
                    "object": "person",
                    "count": 62
                },
                {
                    "object": "motorcycle",
                    "count": 13
                },
                {
                    "object": "car",
                    "count": 73
                },
                {
                    "object": "truck",
                    "count": 21
                }
            ]
        }
    elif video == "demo-2.mp4":
        processed_video = cd.get_cloudinary_playback_url("processed_videos/yjsvq4zaukliptxhddir")

        return {
            "status":"success",
             "video_url":processed_video,
             "counts":[{"object":"car","count":212},{"object":"truck","count":58},{"object":"person","count":2},{"object":"motorcycle","count":2}]
        }
    
    elif video == "demo-3.mp4":
        processed_video = cd.get_cloudinary_playback_url("processed_videos/yph9jdkrklvs9b3gv4ia")
        
        return {
            "status": "success",
            "video_url":processed_video,
            "counts": [
                {
                    "object": "person",
                    "count": 170
                },
                {
                    "object": "car",
                    "count": 1
                }
            ]
        }