import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException

from app.detector import detect_and_count_objects
from app.cloudinary_uploader import upload_video

app = FastAPI()

UPLOAD_DIR = "temp/uploads"
OUTPUT_DIR = "temp/outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.post("/process-video")
def process_video(file: UploadFile = File(...)):
    # -----------------------------
    # Validate file type
    # -----------------------------
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid video file")

    video_id = str(uuid.uuid4())

    input_path = os.path.join(UPLOAD_DIR, f"{video_id}_{file.filename}")
    output_path = os.path.join(OUTPUT_DIR, f"{video_id}_output.mp4")

    # -----------------------------
    # Save uploaded video (SAFE)
    # -----------------------------
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # -----------------------------
    # Run YOLO detection ONCE
    # -----------------------------
    counts = detect_and_count_objects(
        video_path=input_path,
        output_video_path=output_path,
        model_path="yolo11l.pt"   # choose ONE model
    )

    # -----------------------------
    # Upload to Cloudinary
    # -----------------------------
    video_url = upload_video(output_path)

    return {
        "status": "success",
        "video_url": video_url,
        "counts": counts
    }
