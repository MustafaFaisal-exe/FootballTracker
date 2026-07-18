import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "reconnect;1|reconnect_streamed;1|reconnect_delay_max;5|timeout;15000000"
)

YOUTUBE_URL = "https://www.youtube.com/watch?v=f8xrN8Go4kQ&pp=ygUOZmlmYSB3b3JsZCBjdXA%3D"
MODEL_PATH = "models/yolo26n.onnx"
CONF_THRESHOLD = 0.30
DETECT_EVERY_N_FRAMES = 4
DISPLAY_WIDTH = 1280
MAX_STREAM_HEIGHT = 1080

QUALITY_OPTIONS = [
    ("360p", 360),
    ("480p", 480),
    ("720p", 720),
    ("1080p", 1080),
    ("1440p", 1440),
    ("2160p (4K)", 2160),
]