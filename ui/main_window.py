import cv2
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.pitch_view import PitchView
from config import YOUTUBE_URL, QUALITY_OPTIONS, MAX_STREAM_HEIGHT
from ui.worker import TrackerWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Football Tracker")
        self.resize(1280, 800)

        self.worker: TrackerWorker | None = None

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Paste a YouTube URL...")
        self.url_input.setText(YOUTUBE_URL)

        # --- quality dropdown ---
        self.quality_combo = QComboBox()
        for label, height in QUALITY_OPTIONS:
            self.quality_combo.addItem(label, userData=height)
        default_index = next(
            (i for i, (_, h) in enumerate(QUALITY_OPTIONS) if h == MAX_STREAM_HEIGHT), 0
        )
        self.quality_combo.setCurrentIndex(default_index)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_tracking)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_tracking)
        self.stop_btn.setEnabled(False)

        self.calibrate_btn = QPushButton("Detect Team Colors")
        self.calibrate_btn.setEnabled(False)
        self.calibrate_btn.clicked.connect(self.request_calibration)

        self.team_a_btn = QPushButton("Team A")
        self.team_b_btn = QPushButton("Team B")
        self.team_a_btn.setEnabled(False)
        self.team_b_btn.setEnabled(False)
        self.team_a_btn.clicked.connect(lambda: self.select_team("A"))
        self.team_b_btn.clicked.connect(lambda: self.select_team("B"))
        self._set_team_button_style("A", "#ffff00", selected=True)
        self._set_team_button_style("B", "#ff00ff", selected=False)

        self.video_label = QLabel("No video loaded")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #111; color: #888;")
        self.video_label.setMinimumSize(960, 540)

        self.status_label = QLabel("Idle")

        self.pitch_view = PitchView(
            "assets/pitch.png",
            width=250,
            height=160,
            keep_aspect=True,
            parent=self.video_label,
        )
        self._reposition_pitch_view()
        self.pitch_view.hide()

        controls = QHBoxLayout()
        controls.addWidget(self.url_input)
        controls.addWidget(self.quality_combo)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        controls.addWidget(self.calibrate_btn)
        controls.addWidget(self.team_a_btn)
        controls.addWidget(self.team_b_btn)

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(self.video_label, stretch=1)
        layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def _reposition_pitch_view(self):
        label_w = self.video_label.width()
        label_h = self.video_label.height()
        pitch_w = self.pitch_view.width()
        pitch_h = self.pitch_view.height()

        x = (label_w - pitch_w) // 2
        y = label_h - pitch_h - 16
        self.pitch_view.move(x, y)
        self.pitch_view.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_pitch_view()

    def start_tracking(self):
        url = self.url_input.text().strip()
        if not url:
            self.status_label.setText("Enter a YouTube URL first.")
            return

        max_height = self.quality_combo.currentData()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.url_input.setEnabled(False)
        self.quality_combo.setEnabled(False)

        self.worker = TrackerWorker(url, max_height=max_height)
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.status.connect(self.on_status)
        self.worker.team_colors_ready.connect(self.on_team_colors_ready)
        self.worker.calibration_state.connect(self.on_calibration_state)
        self.worker.error.connect(self.on_error)
        self.worker.finished_playing.connect(self.on_finished)
        self.calibrate_btn.setEnabled(True)
        self.worker.start()

    def request_calibration(self):
        if self.worker is not None and self.worker.isRunning():
            self.calibrate_btn.setEnabled(False)
            self.team_a_btn.setEnabled(False)
            self.team_b_btn.setEnabled(False)
            self.worker.request_calibration()

    @Slot(bool)
    def on_calibration_state(self, active):
        self.calibrate_btn.setEnabled(not active)

    def select_team(self, team):
        if self.worker is not None:
            self.worker.set_selected_team(team)
        self._set_team_button_style("A", self.team_a_btn.property("team_color") or "#ffff00", team == "A")
        self._set_team_button_style("B", self.team_b_btn.property("team_color") or "#ff00ff", team == "B")

    def _set_team_button_style(self, team, color, selected=False):
        button = self.team_a_btn if team == "A" else self.team_b_btn
        button.setProperty("team_color", color)
        border = "3px solid white" if selected else "1px solid #555"
        button.setStyleSheet(f"QPushButton {{ background-color: {color}; color: #111; border: {border}; padding: 4px 10px; }}")

    def stop_tracking(self):
        if self.worker is not None:
            self.worker.stop()

    @Slot(object, float, int)
    def on_frame_ready(self, frame_bgr, fps, track_count):
        self.pitch_view.show()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)
        self.status_label.setText(f"{fps:.1f} FPS  |  {track_count} active tracks")

        self._reposition_pitch_view()

    @Slot(object)
    def on_team_colors_ready(self, colors):
        if not colors:
            return
        for team, color in colors.items():
            b, g, r = (max(0, min(255, int(v))) for v in color)
            self._set_team_button_style(team, f"rgb({r}, {g}, {b})", team == "A")
        self.team_a_btn.setEnabled("A" in colors)
        self.team_b_btn.setEnabled("B" in colors)

    def on_status(self, message):
        self.status_label.setText(message)

    @Slot(str)
    def on_error(self, message):
        self.status_label.setText(f"Error: {message}")

    @Slot()
    def on_finished(self):
        self.pitch_view.hide()

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.url_input.setEnabled(True)
        self.quality_combo.setEnabled(True)
        self.team_a_btn.setEnabled(False)
        self.team_b_btn.setEnabled(False)
        self.calibrate_btn.setEnabled(False)

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        event.accept()