import time

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from config import MODEL_PATH, DETECT_EVERY_N_FRAMES, MAX_STREAM_HEIGHT
from tracker import ByteTracker, draw_tracks, extract_detections, get_stream_url, load_session

class TrackerWorker(QThread):
    frame_ready = Signal(np.ndarray, float, int)
    status = Signal(str)
    error = Signal(str)
    finished_playing = Signal()

    def __init__(self, youtube_url: str, max_height: int = MAX_STREAM_HEIGHT, parent=None):
        super().__init__(parent)
        self.youtube_url = youtube_url
        self.max_height = max_height
        self._running = False

    def stop(self):
        self._running = False

    def run(self):
        try:
            self.status.emit(f"Resolving stream URL ({self.max_height}p)...")
            stream_url = get_stream_url(self.youtube_url, max_height=self.max_height)
            # ...rest unchanged

            self.status.emit("Opening video stream...")
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                self.error.emit("Could not open video stream. URL may have expired or is unreachable.")
                return

            self.status.emit("Loading YOLO model...")
            session = load_session(MODEL_PATH)
            input_name = session.get_inputs()[0].name
            output_names = [o.name for o in session.get_outputs()]

            blank_warmup = np.zeros((1, 3, 640, 640), dtype=np.float32)
            session.run(output_names, {input_name: blank_warmup})

            tracker = ByteTracker()
            frame_idx = 0
            self._running = True
            self.status.emit("Playing")

            while self._running:
                loop_t0 = time.perf_counter()

                ret, frame = cap.read()
                if not ret:
                    self.status.emit("Stream ended or read failed.")
                    break

                frame_idx += 1
                tracker.predict()

                if frame_idx % DETECT_EVERY_N_FRAMES == 0:
                    detections, _ = extract_detections(frame, session, input_name, output_names)
                    tracker.update(detections)

                annotated = draw_tracks(frame, tracker.get_active_tracks())

                total_ms = (time.perf_counter() - loop_t0) * 1000
                fps = 1000.0 / total_ms if total_ms > 0 else 0.0

                self.frame_ready.emit(annotated, fps, len(tracker.tracks))

            cap.release()
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished_playing.emit()