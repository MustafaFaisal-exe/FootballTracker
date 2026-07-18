from .kalman import Kalman1D

_track_id_counter = 0


class Track:
    def __init__(self, box, label, score, color=None):
        global _track_id_counter
        _track_id_counter += 1
        self.id = _track_id_counter
        self.label = label
        self.score = score
        self.color = color

        x, y, w, h = box
        cx, cy = x + w / 2, y + h / 2

        self.kx = Kalman1D(cx, 1, 25)
        self.ky = Kalman1D(cy, 1, 25)
        self.kw = Kalman1D(w, 0.5, 50)
        self.kh = Kalman1D(h, 0.5, 50)

        self.hits = 1
        self.age = 0
        self.time_since_update = 0

    def predict(self):
        self.kx.predict()
        self.ky.predict()
        self.kw.predict()
        self.kh.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, box, score, color=None):
        x, y, w, h = box
        cx, cy = x + w / 2, y + h / 2
        self.kx.update(cx)
        self.ky.update(cy)
        self.kw.update(w)
        self.kh.update(h)
        self.score = score
        if color is not None:
            self.color = color
        self.hits += 1
        self.time_since_update = 0

    def get_box(self):
        w = max(0.0, self.kw.x)
        h = max(0.0, self.kh.x)
        return (self.kx.x - w / 2, self.ky.x - h / 2, w, h)

    @property
    def feet(self):
        x, y, w, h = self.get_box()
        return (int(x + w / 2), int(y + h))

    @property
    def vx(self):
        return self.kx.v

    @property
    def vy(self):
        return self.ky.v


def iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return 0.0 if union <= 0 else inter / union


def greedy_match(tracks, dets, iou_threshold):
    pairs = []
    for ti, t in enumerate(tracks):
        for di, d in enumerate(dets):
            if t.label != d["label"]:
                continue
            val = iou(t.get_box(), d["box"])
            if val >= iou_threshold:
                pairs.append((val, ti, di))

    pairs.sort(key=lambda p: p[0], reverse=True)

    matched_tracks, matched_dets = set(), set()
    matches = []
    for _, ti, di in pairs:
        if ti in matched_tracks or di in matched_dets:
            continue
        matches.append((ti, di))
        matched_tracks.add(ti)
        matched_dets.add(di)

    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
    unmatched_dets = [i for i in range(len(dets)) if i not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets