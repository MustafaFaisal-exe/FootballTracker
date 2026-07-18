from config import CONF_THRESHOLD, DETECT_EVERY_N_FRAMES
from .track import Track, greedy_match


class ByteTracker:
    def __init__(self, high_thresh=0.5, low_thresh=CONF_THRESHOLD, iou_threshold=0.3,
                 max_age=DETECT_EVERY_N_FRAMES * 3):
        self.tracks = []
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    def predict(self):
        for t in self.tracks:
            t.predict()

    def update(self, detections, frames_stale=0):
        high_dets = [d for d in detections if d["score"] >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d["score"] < self.high_thresh]

        def compensate(track, box):
            x, y, w, h = box
            return (x + track.vx * frames_stale, y + track.vy * frames_stale, w, h)

        matches, unmatched_tracks, unmatched_dets = greedy_match(self.tracks, high_dets, self.iou_threshold)
        for ti, di in matches:
            track = self.tracks[ti]
            d = high_dets[di]
            track.update(compensate(track, d["box"]), d["score"], d.get("color"))

        remaining_tracks = [self.tracks[i] for i in unmatched_tracks]
        matches2, _, _ = greedy_match(remaining_tracks, low_dets, self.iou_threshold)
        for ti, di in matches2:
            track = remaining_tracks[ti]
            d = low_dets[di]
            track.update(compensate(track, d["box"]), d["score"], d.get("color"))

        for di in unmatched_dets:
            d = high_dets[di]
            self.tracks.append(Track(d["box"], d["label"], d["score"], d.get("color")))

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

    def get_active_tracks(self, max_time_since_update=None):
        if max_time_since_update is None:
            max_time_since_update = DETECT_EVERY_N_FRAMES + 6
        return [t for t in self.tracks if t.time_since_update <= max_time_since_update]