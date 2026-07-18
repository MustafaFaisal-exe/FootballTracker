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

        self.video_label = QLabel("No video loaded")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #111; color: #888;")
        self.video_label.setMinimumSize(960, 540)

        self.status_label = QLabel("Idle")

        controls = QHBoxLayout()
        controls.addWidget(self.url_input)
        controls.addWidget(self.quality_combo)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(self.video_label, stretch=1)
        layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

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
        self.worker.error.connect(self.on_error)
        self.worker.finished_playing.connect(self.on_finished)
        self.worker.start()

    def stop_tracking(self):
        if self.worker is not None:
            self.worker.stop()

    @Slot(object, float, int)
    def on_frame_ready(self, frame_bgr, fps, track_count):
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

    @Slot(str)
    def on_status(self, message):
        self.status_label.setText(message)

    @Slot(str)
    def on_error(self, message):
        self.status_label.setText(f"Error: {message}")

    @Slot()
    def on_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.url_input.setEnabled(True)
        self.quality_combo.setEnabled(True)

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        event.accept()