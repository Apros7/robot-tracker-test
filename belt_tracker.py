"""Track objects on a belt and predict where each one will be.

Pixels become belt millimetres through a homography from four markers. Every
object shares one speed along the belt. Position at a future time is the last
known position plus that speed times the elapsed time.

The marker points below are placeholders. Replace PIXEL_PTS and BELT_MM with
measurements before the millimetres, or the pick line, mean anything physical.
"""

from __future__ import annotations

import itertools
import os
import threading
from collections import deque

import cv2
import numpy as np

MODEL_ID = "easysort/elongated-metal-object-trial-1-rfdetr-nano-t1"
VIDEO_SOURCE = 0
CAMERA_FPS = 30

# Pixel positions of flat markers, and where those markers sit on the belt in
# millimetres. x_mm points in the direction of travel.
PIXEL_PTS = np.array([[412, 210], [1508, 205], [1520, 880], [400, 890]], dtype=np.float32)
BELT_MM = np.array([[0, 0], [600, 0], [600, 400], [0, 400]], dtype=np.float32)
H, _ = cv2.findHomography(PIXEL_PTS, BELT_MM)

CAMERA_LATENCY_S = 0.0
PICK_LINE_MM = 900.0
MIN_SAMPLES = 5
MATCH_GATE_MM = 40.0
FORGET_AFTER_S = 3.0
# A detection has to be the same object as last frame, and actually moving,
# before it can take the single prediction away from a stationary one.
ASSOCIATE_MM = 150.0
MOVE_MM_S = 60.0
STILL_MM_S = 40.0
# Every sighting stays in the fit, so a long look keeps tightening the estimate.
MAX_SAMPLES = 20000

history: dict[int, deque] = {}
_next_id = itertools.count(1)
belt_speed: float | None = None
previous_mm: np.ndarray | None = None
previous_t: float | None = None
lock = threading.Lock()


def reset() -> None:
    """Clear tracks and the speed estimate. Used by tests and each live run."""
    global belt_speed, _next_id, previous_mm, previous_t
    with lock:
        history.clear()
        belt_speed = None
        previous_mm = None
        previous_t = None
        _next_id = itertools.count(1)


def to_belt_mm(points_px: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
    if len(pts) == 0:
        return np.zeros((0, 2), np.float32)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def to_pixels(points_mm: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_mm, dtype=np.float32).reshape(-1, 1, 2)
    if len(pts) == 0:
        return np.zeros((0, 2), np.float32)
    return cv2.perspectiveTransform(pts, np.linalg.inv(H)).reshape(-1, 2)


def detection_centres(predictions):
    """Pixel centres from an Inference result (dict or response object)."""
    preds = predictions.get("predictions", []) if isinstance(predictions, dict) else getattr(predictions, "predictions", [])
    out = []
    for pred in preds:
        x = pred["x"] if isinstance(pred, dict) else pred.x
        y = pred["y"] if isinstance(pred, dict) else pred.y
        out.append((x, y))
    return np.array(out, dtype=np.float32).reshape(-1, 2)


def track_slope(samples):
    """Least-squares velocity (mm/s) along the belt for one track."""
    times = np.array([sample[0] for sample in samples])
    positions = np.array([sample[1] for sample in samples])
    if len(times) < MIN_SAMPLES or times[-1] - times[0] < 0.1:
        return None
    velocity, _ = np.polyfit(times - times[0], positions, 1)
    return float(velocity)


def _nearest_previous(point: np.ndarray) -> np.ndarray | None:
    if previous_mm is None or len(previous_mm) == 0:
        return None
    deltas = np.hypot(previous_mm[:, 0] - point[0], previous_mm[:, 1] - point[1])
    index = int(np.argmin(deltas))
    if float(deltas[index]) > ASSOCIATE_MM:
        return None
    return previous_mm[index]


def _point_speed(t: float, point: np.ndarray) -> float:
    prev = _nearest_previous(point)
    if prev is None or previous_t is None or t <= previous_t:
        return 0.0
    return float(np.hypot(point[0] - prev[0], point[1] - prev[1])) / (t - previous_t)


def _mover(t: float, points_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """The detection that moved since the last frame, if one clearly did."""
    if previous_t is None or t <= previous_t or len(points_mm) == 0:
        return None
    best = None
    for point in points_mm:
        prev = _nearest_previous(point)
        speed = _point_speed(t, point)
        if prev is not None and speed >= MOVE_MM_S and (best is None or speed > best[0]):
            best = (speed, point, prev)
    if best is None:
        return None
    return best[1], best[2]


def _start(t: float, point: np.ndarray, prev: np.ndarray | None) -> None:
    """Begin the one prediction on this object, seeding it with where it just was."""
    global belt_speed
    history.clear()
    belt_speed = None
    samples: deque = deque(maxlen=MAX_SAMPLES)
    if prev is not None and previous_t is not None and t > previous_t:
        samples.append((previous_t, float(prev[0]), float(prev[1])))
    samples.append((t, float(point[0]), float(point[1])))
    history[next(_next_id)] = samples


def _remember(t: float, points_mm: np.ndarray) -> None:
    global previous_mm, previous_t
    previous_mm = np.array(points_mm, np.float32).reshape(-1, 2)
    previous_t = t


def _match(t, points_mm):
    """Follow the object that is moving. A stationary detection does not get the prediction.

    Once the moving object leaves view, the same prediction keeps coasting. Another
    object that is sitting still cannot take it over during that time.
    """
    points_mm = np.asarray(points_mm, dtype=np.float32).reshape(-1, 2)
    mover = _mover(t, points_mm)
    if not history:
        if mover is not None:
            _start(t, mover[0], mover[1])
        _remember(t, points_mm)
        return

    samples = next(iter(history.values()))
    t_last, x_last, y_last = samples[-1]
    velocity = belt_speed or 0.0
    gate = MATCH_GATE_MM if belt_speed else 3 * MATCH_GATE_MM
    x_pred = x_last + velocity * (t - t_last)
    best = None
    for x_mm, y_mm in points_mm:
        distance = float(np.hypot(x_mm - x_pred, y_mm - y_last))
        if distance < gate and (best is None or distance < best[0]):
            best = (distance, float(x_mm), float(y_mm))

    still = belt_speed is None or abs(belt_speed) < STILL_MM_S
    best_point = None if best is None else np.array([best[1], best[2]], np.float32)
    best_is_moving = best_point is not None and _point_speed(t, best_point) >= MOVE_MM_S
    mover_is_other = mover is not None and (best is None or float(np.hypot(*(mover[0] - best_point))) > 1)
    if mover is not None and still and mover_is_other and not best_is_moving:
        _start(t, mover[0], mover[1])
    elif best is not None:
        samples.append((t, best[1], best[2]))
    _remember(t, points_mm)


def update(t, centres_px):
    """Feed one frame: timestamp (s) and detection centres in pixels."""
    global belt_speed
    centres_px = np.asarray(centres_px, dtype=np.float32).reshape(-1, 2)
    with lock:
        if len(centres_px):
            _match(t, to_belt_mm(centres_px))
        elif not history:
            return

        if history:
            slope = track_slope(next(iter(history.values())))
            if slope is not None:
                belt_speed = slope

        for track_id in [key for key, samples in history.items() if t - samples[-1][0] > FORGET_AFTER_S]:
            del history[track_id]
        if not history:
            belt_speed = None


def _reference(track_id):
    """Noise-reduced (t_ref, x_ref, y) for a track using the shared belt speed."""
    samples = history[track_id]
    times = np.array([sample[0] for sample in samples])
    positions = np.array([sample[1] for sample in samples])
    sideways = np.array([sample[2] for sample in samples])
    t_ref = float(times[-1])
    x_ref = float(np.mean(positions + belt_speed * (t_ref - times)))
    return t_ref, x_ref, float(np.median(sideways))


def predict_position(track_id, t_query):
    """Belt position (x_mm, y_mm) of object `track_id` at absolute time `t_query`."""
    with lock:
        if belt_speed is None or len(history.get(track_id, ())) < MIN_SAMPLES:
            return None
        t_ref, x_ref, y_mm = _reference(track_id)
        return x_ref + belt_speed * (t_query - t_ref), y_mm


def pick_plan():
    """For every trusted object upstream of the pick line: when it arrives and where (y)."""
    plan = {}
    with lock:
        if not belt_speed or belt_speed <= 0:
            return plan
        for track_id, samples in history.items():
            if len(samples) < MIN_SAMPLES:
                continue
            t_ref, x_ref, y_mm = _reference(track_id)
            if x_ref < PICK_LINE_MM:
                plan[track_id] = {"t_pick": t_ref + (PICK_LINE_MM - x_ref) / belt_speed, "y_mm": y_mm}
    return plan


def snapshot(t_now: float) -> dict:
    """Tracks, shared speed, and where each trusted object will be."""
    with lock:
        speed = belt_speed
        tracks = []
        for track_id, samples in history.items():
            t_last, x_last, y_last = samples[-1]
            trusted = speed is not None and len(samples) >= MIN_SAMPLES
            points = []
            if trusted:
                t_ref, x_ref, y_mm = _reference(track_id)
                points.append({
                    "t": t_now,
                    "dt": 0.0,
                    "x_mm": x_ref + speed * (t_now - t_ref),
                    "y_mm": y_mm,
                })
            tracks.append({
                "id": track_id,
                "t": t_last,
                "seen": abs(t_last - t_now) < 1e-6,
                "age": t_now - t_last,
                "seen_for": t_last - samples[0][0],
                "x_mm": x_last,
                "y_mm": y_last,
                "trusted": trusted,
                "points": points,
            })
        plan = {}
        if speed and speed > 0:
            for track in tracks:
                if not track["trusted"]:
                    continue
                current = track["points"][0]
                if current["x_mm"] < PICK_LINE_MM:
                    plan[track["id"]] = {
                        "t_pick": t_now + (PICK_LINE_MM - current["x_mm"]) / speed,
                        "y_mm": current["y_mm"],
                    }
        return {"speed": speed, "tracks": tracks, "plan": plan}


def _inside(point, shape) -> bool:
    height, width = shape[:2]
    x_px, y_px = point
    return -30 <= x_px < width + 30 and -30 <= y_px < height + 30


def annotate(image: np.ndarray, t: float, preds: list) -> dict:
    """Update the tracker from this frame and draw its one predicted position."""
    ordered = sorted(preds, key=lambda pred: pred.get("confidence", 0), reverse=True)
    centres = np.array([[pred["x"], pred["y"]] for pred in ordered], np.float32) if ordered else np.zeros((0, 2), np.float32)
    update(t, centres)
    info = snapshot(t)
    _draw_belt_axis(image)
    names = _names_for_tracks(info["tracks"], preds)
    for track in info["tracks"]:
        _draw_track(image, track, info["plan"].get(track["id"]), names.get(track["id"]))
    speed = info["speed"]
    label = "belt speed: learning" if speed is None else f"belt {speed:.1f} mm/s"
    cv2.putText(image, label, (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, label, (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)
    return info


def _names_for_tracks(tracks: list, preds: list) -> dict:
    names = {}
    if not preds:
        return names
    centres = np.array([[pred["x"], pred["y"]] for pred in preds], np.float32)
    for track in tracks:
        if not track["seen"]:
            continue
        pixel = to_pixels([[track["x_mm"], track["y_mm"]]])[0]
        nearest = int(np.argmin(np.linalg.norm(centres - pixel, axis=1)))
        if np.linalg.norm(centres[nearest] - pixel) < 20:
            names[track["id"]] = preds[nearest].get("class") or preds[nearest].get("class_name") or ""
    return names


def _draw_belt_axis(image: np.ndarray) -> None:
    direction = to_pixels([[200.0, 200.0]])[0] - to_pixels([[0.0, 200.0]])[0]
    length = float(np.hypot(*direction))
    if length < 1:
        return
    direction = direction / length * 90
    origin = np.array([48.0, image.shape[0] - 48.0])
    end = origin + direction
    cv2.arrowedLine(image, tuple(np.int32(origin)), tuple(np.int32(end)), (255, 255, 255), 2, tipLength=0.3)
    cv2.putText(image, "belt", tuple(np.int32(end + np.array([8, 6]))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_track(image: np.ndarray, track: dict, plan: dict | None, name: str | None) -> None:
    if not track["points"]:
        if not track["seen"]:
            return
        pixel = to_pixels([[track["x_mm"], track["y_mm"]]])[0]
        cv2.putText(image, "learning", (int(pixel[0]) + 8, int(pixel[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        return
    pixel = to_pixels([[track["points"][0]["x_mm"], track["points"][0]["y_mm"]]])[0]
    if not _inside(pixel, image.shape):
        return
    center = (int(round(pixel[0])), int(round(pixel[1])))
    cv2.circle(image, center, 8, (0, 170, 255), 2, cv2.LINE_AA)
    if track["seen"]:
        label = name or "object"
        label = f"{label}  {track['seen_for']:.1f}s"
    else:
        label = f"predicted {track['age']:.1f}s"
    if plan and track["seen"]:
        label = f"{label}  pick {plan['t_pick'] - track['points'][0]['t']:.1f}s"
    cv2.putText(image, label, (center[0] + 12, center[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, label, (center[0] + 12, center[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 170, 255), 1, cv2.LINE_AA)


def on_prediction(predictions, video_frame):
    t = video_frame.frame_timestamp.timestamp() - CAMERA_LATENCY_S
    update(t, detection_centres(predictions))
    for track_id, info in pick_plan().items():
        print(
            f"id {track_id}: y={info['y_mm']:.0f} mm, reaches pick line at t={info['t_pick']:.3f} "
            f"(in {info['t_pick'] - t:.2f} s), belt {belt_speed:.1f} mm/s"
        )


if __name__ == "__main__":
    from inference import InferencePipeline

    pipeline = InferencePipeline.init(
        model_id=MODEL_ID,
        api_key=os.environ["ROBOFLOW_API_KEY"],
        video_reference=VIDEO_SOURCE,
        on_prediction=on_prediction,
        max_fps=CAMERA_FPS,
    )
    pipeline.start()
    pipeline.join()
