import cloudinary
import cloudinary.uploader
import cloudinary.api
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
    # chunk_size=6000000,   # 6MB chunks (important)
    # timeout=300           # 5 minutes
)
    # print(result)
    return result["playback_url"]


def get_cloudinary_playback_url(public_id: str) -> str:
    """
    Returns adaptive streaming (HLS) playback URL
    """
    result = cloudinary.api.resource(
        public_id,
        resource_type="video"
    )

    return result["secure_url"].replace(".mp4", ".m3u8")
