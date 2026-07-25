import cv2
import numpy as np

from config import (TEAM_CALIBRATION_DETECTIONS, TEAM_COLOR_DISTANCE_THRESHOLD, TEAM_COLOR_CLUSTERS, TEAM_MIN_PLAYERS)

_KMEANS_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)


def _fmt_bgr(bgr):
    b, g, r = (int(c) for c in bgr)
    return f"RGB({r},{g},{b}) #{r:02x}{g:02x}{b:02x}"


class TeamCalibrator:
    """Learn the two team jersey colors once, then classify/filter players by team."""

    def __init__(
        self,
        n_frames=TEAM_CALIBRATION_DETECTIONS,
        distance_threshold=TEAM_COLOR_DISTANCE_THRESHOLD,
        min_players=TEAM_MIN_PLAYERS,
    ):
        self.n_frames = n_frames
        self.min_players = min_players
        self.distance_threshold = distance_threshold
        self.frame_count = 0
        self.samples = []
        self.team_colors = None
        self.team_color_tolerances = None
        self.calibrated = False

    def is_calibrating(self):
        return not self.calibrated

    def add_detection_pass(self, player_detections):
        if self.calibrated:
            return

        if len(player_detections) < self.min_players:
            return

        for det in player_detections:
            color = det.get("color")
            if color is not None:
                self.samples.append(color)

        self.frame_count += 1
        if self.frame_count >= self.n_frames:
            self._finalize()

    def _finalize(self):
        self.calibrated = True
        if len(self.samples) < 2:
            print("[TEAM COLORS] Not enough player samples to identify teams.")
            return

        colors = np.array(self.samples, dtype=np.float32)
        # A third cluster absorbs referee/outlier colors; retain the two largest.
        cluster_count = min(TEAM_COLOR_CLUSTERS, len(colors))
        _, labels, centers = cv2.kmeans(colors, cluster_count, None, _KMEANS_CRITERIA, attempts=5, flags=cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.flatten(), minlength=cluster_count)
        # Prefer two well-supported, separated groups. This prevents a
        # referee cluster from replacing a team, and avoids selecting two
        # lighting shades from the same kit.
        pairs = [(i, j) for i in range(cluster_count) for j in range(i + 1, cluster_count)]
        def pair_score(pair):
            i, j = pair
            separation = float(np.linalg.norm(centers[i] - centers[j]))
            support = min(int(counts[i]), int(counts[j]))
            return support * separation, support
        selected_pair = max(pairs, key=pair_score)
        self.team_colors = {
            "A": np.array(centers[selected_pair[0]]),
            "B": np.array(centers[selected_pair[1]]),
        }
        self.team_color_tolerances = {}
        for team, cluster_id in zip(("A", "B"), selected_pair):
            member_colors = colors[labels.flatten() == cluster_id]
            distances = np.linalg.norm(member_colors - centers[cluster_id], axis=1)
            # Keep a configurable minimum range, expanded when the observed
            # calibration samples naturally vary due to lighting/compression.
            observed_range = float(np.percentile(distances, 95)) + 10.0
            self.team_color_tolerances[team] = max(self.distance_threshold, observed_range)
        print(
            f"[TEAM COLORS] Calibrated from {self.frame_count} player-detection frames "
            f"({len(self.samples)} player samples):"
        )
        print(f"  Team A: {_fmt_bgr(self.team_colors['A'])}")
        print(f"  Team B: {_fmt_bgr(self.team_colors['B'])}")

    def classify(self, color):
        if not self.calibrated or self.team_colors is None or color is None:
            return None

        distances = {
            team: float(np.linalg.norm(color - center))
            for team, center in self.team_colors.items()
        }
        valid = [
            team for team, distance in distances.items()
            if distance <= self.team_color_tolerances.get(team, self.distance_threshold)
        ]
        if not valid:
            return None
        return min(valid, key=distances.get)

    def apply(self, detections, track_teams):
        if not self.calibrated:
            return detections

        filtered = []
        for det in detections:
            if det["label"] != "player":
                filtered.append(det)
                continue

            team = self.classify(det.get("color"))
            if team is None or team not in track_teams:
                continue

            det = dict(det)
            det["team"] = team
            filtered.append(det)
        return filtered
