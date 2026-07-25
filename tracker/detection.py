import time

import cv2
import numpy as np
import onnxruntime as ort

from config import CONF_THRESHOLD

LOWER_GREEN = np.array([35, 40, 40])
UPPER_GREEN = np.array([85, 255, 255])


def _torso_pixels(frame, box, hsv_full):
    x1, y1, w, h = box
    hh, ww = h, w
    if hh <= 0 or ww <= 0:
        return None
    ty1, ty2 = y1 + int(hh * 0.25), y1 + int(hh * 0.55)
    tx1, tx2 = x1 + int(ww * 0.25), x1 + int(ww * 0.75)
    bgr_torso = frame[ty1:ty2, tx1:tx2]
    hsv_torso = hsv_full[ty1:ty2, tx1:tx2]
    if bgr_torso.size == 0:
        return None
    green_mask = cv2.inRange(hsv_torso, LOWER_GREEN, UPPER_GREEN)
    non_green = bgr_torso[green_mask == 0]
    return non_green if len(non_green) >= 5 else bgr_torso.reshape(-1, 3)


def extract_player_color(frame, box, hsv_full):
    pixels = _torso_pixels(frame, box, hsv_full)
    if pixels is None or len(pixels) == 0:
        return np.array([0.0, 0.0, 0.0])
    # The median is less affected by skin, shorts, and pitch pixels than a
    # mean, so the sample better represents the jersey itself.
    return np.median(pixels, axis=0)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_NEAREST)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def load_session(model_path: str) -> ort.InferenceSession:
    available_providers = ort.get_available_providers()
    if "OpenVINOExecutionProvider" in available_providers:
        target_providers = [("OpenVINOExecutionProvider", {"device_type": "CPU"})]
        print("[SYSTEM INFO]: Intel OpenVINO Execution Engine successfully hooked and active.")
    else:
        target_providers = ["CPUExecutionProvider"]
        print("[SYSTEM WARNING]: OpenVINO provider not detected. Falling back to native CPU runtime.")
        print("  -> Fix: pip uninstall onnxruntime -y && pip install onnxruntime-openvino")

    return ort.InferenceSession(model_path, providers=target_providers)


def extract_detections(frame, session, input_name, output_names):
    h_orig, w_orig = frame.shape[:2]

    img_padded, ratio, (dw, dh) = letterbox(frame, (640, 640))
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_data = np.transpose(img_rgb.astype(np.float32) / 255.0, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)

    t0 = time.perf_counter()
    outputs = session.run(output_names, {input_name: img_data})
    yolo_time_ms = (time.perf_counter() - t0) * 1000

    predictions = np.squeeze(outputs[0])
    hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    detections = []
    for pred in predictions:
        x1_raw, y1_raw, x2_raw, y2_raw, confidence, class_id = pred
        if confidence <= CONF_THRESHOLD:
            continue
        class_id = int(class_id)
        x1 = max(0, int((x1_raw - dw) / ratio))
        y1 = max(0, int((y1_raw - dh) / ratio))
        x2 = min(w_orig, int((x2_raw - dw) / ratio))
        y2 = min(h_orig, int((y2_raw - dh) / ratio))

        box = (x1, y1, x2 - x1, y2 - y1)

        if class_id == 32:
            detections.append({"box": box, "label": "ball", "score": float(confidence), "color": None})
            continue

        if class_id != 0:
            continue

        color = extract_player_color(frame, box, hsv_full)
        detections.append({
            "box": box,
            "label": "player",
            "score": float(confidence),
            "color": color,
        })

    return detections, yolo_time_ms