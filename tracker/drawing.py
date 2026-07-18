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


def draw_tracks(frame, tracks):
    annotated = frame.copy()

    player_tracks = [t for t in tracks if t.label == "player" and t.color is not None]
    ball_tracks = [t for t in tracks if t.label == "ball"]

    if len(player_tracks) >= 2:
        colors_array = np.array([t.color for t in player_tracks], dtype=np.float32)
        k = min(3, len(player_tracks))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        _, labels, _ = cv2.kmeans(colors_array, k, None, criteria, attempts=3, flags=cv2.KMEANS_PP_CENTERS)
        labels = labels.flatten()

        cluster_counts = np.bincount(labels, minlength=k)
        sorted_clusters = np.argsort(cluster_counts)[::-1]

        team_A = [player_tracks[i] for i in range(len(player_tracks)) if labels[i] == sorted_clusters[0]]
        team_B = (
            [player_tracks[i] for i in range(len(player_tracks)) if labels[i] == sorted_clusters[1]]
            if len(sorted_clusters) > 1
            else []
        )

        connect_tactical_lines(annotated, team_A, (255, 255, 0))
        connect_tactical_lines(annotated, team_B, (255, 0, 255))

        for t in team_A:
            _, _, w, _ = t.get_box()
            r = int(w * 0.5)
            cv2.ellipse(annotated, t.feet, (r, int(r * 0.4)), 0, 0, 360, (255, 255, 0), 2, cv2.LINE_4)
        for t in team_B:
            _, _, w, _ = t.get_box()
            r = int(w * 0.5)
            cv2.ellipse(annotated, t.feet, (r, int(r * 0.4)), 0, 0, 360, (255, 0, 255), 2, cv2.LINE_4)

    for t in ball_tracks:
        cv2.circle(annotated, t.feet, 6, (0, 0, 255), -1, cv2.LINE_4)

    return annotated