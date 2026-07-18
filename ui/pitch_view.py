from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QPixmap, QColor, QBrush, QTransform
from PySide6.QtWidgets import QWidget

class PitchView(QWidget):
    PLAYER_RADIUS = 6

    def __init__(
        self,
        image_path: str,
        width: int = 350,
        height: int = 220,
        keep_aspect: bool = True,
        parent=None,
    ):
        super().__init__(parent)

        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        self._original_pixmap = QPixmap(image_path)

        self._original_pixmap = self._original_pixmap.transformed(
            QTransform().rotate(90),
            Qt.SmoothTransformation
        )
        
        if self._original_pixmap.isNull():
            raise FileNotFoundError(f"Could not load pitch image: {image_path}")

        self.keep_aspect = keep_aspect

        self._player_positions = []

        self.set_pitch_size(width, height)

    def set_pitch_size(self, width: int, height: int):
        """Resize the pitch image."""

        aspect = Qt.KeepAspectRatio if self.keep_aspect else Qt.IgnoreAspectRatio

        self._pixmap = self._original_pixmap.scaled(
            width,
            height,
            aspect,
            Qt.SmoothTransformation,
        )

        self.setFixedSize(self._pixmap.size())

        self.update()

    def set_player_positions(self, positions):
        self._player_positions = positions
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.setOpacity(0.4)   # 40% opacity
        painter.drawPixmap(0, 0, self._pixmap)
        painter.setOpacity(1.0)   # Restore opacity for player dots

        self._draw_players(painter)

    def _draw_players(self, painter):
        painter.setPen(Qt.NoPen)

        team_colors = {
            "A": QColor(255, 255, 0),
            "B": QColor(255, 0, 255),
            "ball": QColor(255, 255, 255),
        }

        w = self._pixmap.width()
        h = self._pixmap.height()

        for x_norm, y_norm, team in self._player_positions:

            px = x_norm * w
            py = y_norm * h

            color = team_colors.get(team, QColor(200, 200, 200))
            painter.setBrush(QBrush(color))

            radius = 4 if team == "ball" else self.PLAYER_RADIUS

            painter.drawEllipse(QPointF(px, py), radius, radius)