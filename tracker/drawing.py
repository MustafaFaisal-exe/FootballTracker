import cv2
import numpy as np


def group_by_depth(team_list, max_lines=3):
    n = len(team_list)
    k = min(max_lines, n)
    if k < 2:
        return [team_list]

    xs = np.array([[t.feet[0]] for t in team_list], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(xs, k, None, criteria, attempts=3, flags=cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()

    depth_order = np.argsort(centers.flatten())
    lines = []
    for cluster_id in depth_order:
        line = [team_list[i] for i in range(n) if labels[i] == cluster_id]
        if line:
            lines.append(line)
    return lines


def connect_tactical_lines(annotated, team_list, line_color):
    if len(team_list) < 2:
        return
    for line in group_by_depth(team_list):
        if len(line) < 2:
            continue
        line.sort(key=lambda t: t.feet[1])
        for i in range(len(line) - 1):
            cv2.line(annotated, line[i].feet, line[i + 1].feet, line_color, 2, cv2.LINE_4)


def _team_bgr_color(team, team_colors):
    fallback = {"A": (255, 255, 0), "B": (255, 0, 255)}[team]
    if not team_colors or team not in team_colors:
        return fallback
    color = np.asarray(team_colors[team], dtype=np.float32)
    return tuple(int(max(0, min(255, channel))) for channel in color)


def draw_tracks(frame, tracks, selected_team=None, team_colors=None):
    annotated = frame.copy()

    team_tracks = {
        "A": [t for t in tracks if t.label == "player" and t.team == "A"],
        "B": [t for t in tracks if t.label == "player" and t.team == "B"],
    }
    ball_tracks = [t for t in tracks if t.label == "ball"]

    teams_to_draw = ("A", "B") if selected_team is None else (selected_team,)
    for team in teams_to_draw:
        if team not in team_tracks:
            continue
        color = _team_bgr_color(team, team_colors)
        players = team_tracks[team]
        connect_tactical_lines(annotated, players, color)
        for track in players:
            _, _, width, _ = track.get_box()
            radius = int(width * 0.5)
            cv2.ellipse(
                annotated,
                track.feet,
                (radius, int(radius * 0.4)),
                0,
                0,
                360,
                color,
                2,
                cv2.LINE_4,
            )

    for track in ball_tracks:
        cv2.circle(annotated, track.feet, 6, (0, 0, 255), -1, cv2.LINE_4)

    return annotated
