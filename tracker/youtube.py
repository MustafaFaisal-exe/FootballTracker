import yt_dlp
from config import MAX_STREAM_HEIGHT

def get_stream_url(youtube_url: str, max_height: int = MAX_STREAM_HEIGHT) -> str:
    ydl_opts = {
        "format": (
            f"bestvideo[height<={max_height}][vcodec!*=av01]"
            f"/bestvideo[height<={max_height}]"
            f"/best[height<={max_height}]"
            f"/best"
        ),
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        if "url" in info:
            return info["url"]
        return info["formats"][-1]["url"]