import time
import threading

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from config import MODEL_PATH, DETECT_EVERY_N_FRAMES, MAX_STREAM_HEIGHT, TRACK_TEAMS
from tracker import ByteTracker, TeamCalibrator, draw_tracks, extract_detections, get_stream_url, load_session

class TrackerWorker(QThread):
    frame_ready = Signal(np.ndarray, float, int)
    status = Signal(str)
    team_colors_ready = Signal(object)
    calibration_state = Signal(bool)
    error = Signal(str)
    finished_playing = Signal()

    def __init__(self, youtube_url: str, max_height: int = MAX_STREAM_HEIGHT, parent=None):
        super().__init__(parent)
        self.youtube_url = youtube_url
        self.max_height = max_height
        self._running = False
        self.selected_team = "A"
        self._calibration_requested = False
        self._calibration_lock = threading.Lock()

    def set_selected_team(self, team):
        if team in TRACK_TEAMS:
            self.selected_team = team

    def request_calibration(self):
        with self._calibration_lock:
            self._calibration_requested = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            self.status.emit(f"Resolving stream URL ({self.max_height}p)...")
            stream_url = get_stream_url(self.youtube_url, max_height=self.max_height)

            self.status.emit("Opening video stream...")
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                self.error.emit("Could not open video stream. URL may have expired or is unreachable.")
                return

            source_fps = cap.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(source_fps) or source_fps <= 1.0:
                source_fps = 30.0
            frame_interval = 1.0 / source_fps
            next_frame_time = time.perf_counter()

            self.status.emit("Loading YOLO model...")
            session = load_session(MODEL_PATH)
            input_name = session.get_inputs()[0].name
            output_names = [o.name for o in session.get_outputs()]

            blank_warmup = np.zeros((1, 3, 640, 640), dtype=np.float32)
            session.run(output_names, {input_name: blank_warmup})

            tracker = ByteTracker()
            team_calibrator = None
            frame_idx = 0
            self._running = True
            self.status.emit("Playing — click Detect Team Colors to assign teams.")

            while self._running:
                loop_t0 = time.perf_counter()

                ret, frame = cap.read()
                if not ret:
                    self.status.emit("Stream ended or read failed.")
                    break

                frame_idx += 1
                tracker.predict()

                with self._calibration_lock:
                    calibration_requested = self._calibration_requested
                    self._calibration_requested = False
                if calibration_requested:
                    team_calibrator = TeamCalibrator()
                    tracker.tracks = []
                    self.calibration_state.emit(True)
                    self.team_colors_ready.emit({})
                    self.status.emit("Calibrating — waiting for 5+ players across 10 detection frames...")

                if frame_idx % DETECT_EVERY_N_FRAMES == 0:
                    detections, _ = extract_detections(frame, session, input_name, output_names)

                    if team_calibrator is not None and team_calibrator.is_calibrating():
                        player_dets = [d for d in detections if d["label"] == "player"]
                        team_calibrator.add_detection_pass(player_dets)
                        if not team_calibrator.is_calibrating():
                            self.team_colors_ready.emit({team: color.tolist() for team, color in team_calibrator.team_colors.items()} if team_calibrator.team_colors else {})
                            self.calibration_state.emit(False)
                            self.status.emit("Playing — team colors detected.")
                            tracker.update(team_calibrator.apply(detections, TRACK_TEAMS))
                        else:
                            ball_dets = [d for d in detections if d["label"] == "ball"]
                            if ball_dets:
                                tracker.update(ball_dets)
                    elif team_calibrator is not None:
                        tracker.update(team_calibrator.apply(detections, TRACK_TEAMS))
                    else:
                        tracker.update(detections)

                annotated = draw_tracks(
                    frame,
                    tracker.get_active_tracks(),
                    selected_team=self.selected_team,
                    team_colors=(team_calibrator.team_colors if team_calibrator and team_calibrator.team_colors else None),
                )

                total_ms = (time.perf_counter() - loop_t0) * 1000
                fps = 1000.0 / total_ms if total_ms > 0 else 0.0

                self.frame_ready.emit(annotated, min(fps, source_fps), len(tracker.tracks))

                # VideoCapture can deliver network frames much faster than their
                # presentation rate. Pace output to the source FPS.
                next_frame_time += frame_interval
                sleep_for = next_frame_time - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)

            cap.release()
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished_playing.emit()