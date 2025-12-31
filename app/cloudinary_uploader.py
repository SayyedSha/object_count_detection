import cloudinary
import cloudinary.uploader
import os
from dotenv import load_dotenv

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)


# def upload_video(video_path):
#     result = cloudinary.uploader.upload(
#         video_path,
#         resource_type="video",
#         folder="processed_videos"
#     )
def upload_video(video_path):
    result = cloudinary.uploader.upload(
    video_path,
    resource_type="video",
    folder="processed_videos",
    chunk_size=6000000,   # 6MB chunks (important)
    timeout=300           # 5 minutes
)
    return result["playback_url"]
