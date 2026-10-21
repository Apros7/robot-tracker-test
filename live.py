"""Live cutlery detector and belt predictor.

Default detector is YOLOE-26l prompted with fork, knife, spoon, and cutlery.
The camera is on the left. One prediction follows the object that is moving and,
once it leaves the frame, continues across the black panel on the right for 3 seconds.

`--detector rfdetr` loads the private Roboflow model instead and needs ROBOFLOW_API_KEY.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import belt_tracker
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_ID = "easysort/elongated-metal-object-trial-1-rfdetr-nano-t1"
VERSION_ID = "elongated-metal-object-trial/1"
YOLOE_WEIGHTS = ROOT / "models" / "yoloe-26l-seg.pt"
CUTLERY_PROMPTS = ["fork", "knife", "spoon", "cutlery"]
HOST = "127.0.0.1"
PORT = 8770

_lock = threading.Lock()
_jpeg: bytes | None = None
_status = {
    "ready": False,
    "message": "Starting",
    "detections": 0,
    "infer_ms": 0.0,
    "fps": 0.0,
    "confidence": 0.40,
    "source": "",
    "model_id": MODEL_ID,
    "title": "Live",
    "note": "",
    "belt_speed_mm_s": None,
    "tracks": 0,
}


def load_env() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def prediction_fields(pred) -> tuple[float, float, float, float, float, str]:
    if isinstance(pred, dict):
        name = pred.get("class") or pred.get("class_name") or "elongated metal object"
        return float(pred["x"]), float(pred["y"]), float(pred["width"]), float(pred["height"]), float(pred["confidence"]), str(name)
    name = getattr(pred, "class_name", None) or getattr(pred, "class", None) or "elongated metal object"
    return float(pred.x), float(pred.y), float(pred.width), float(pred.height), float(pred.confidence), str(name)


def predictions_of(result) -> list:
    if isinstance(result, list):
        result = result[0] if result else {}
    preds = getattr(result, "predictions", None)
    if preds is None and isinstance(result, dict):
        preds = result.get("predictions", [])
    return list(preds or [])


def draw(frame: np.ndarray, preds: list, infer_ms: float) -> np.ndarray:
    image = frame.copy()
    for pred in preds:
        cx, cy, w, h, score, name = prediction_fields(pred)
        x1 = int(round(cx - w / 2))
        y1 = int(round(cy - h / 2))
        x2 = int(round(cx + w / 2))
        y2 = int(round(cy + h / 2))
        cv2.rectangle(image, (x1, y1), (x2, y2), (80, 220, 80), 2)
        label = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        top = max(0, y1 - th - 8)
        cv2.rectangle(image, (x1, top), (x1 + tw + 8, top + th + 8), (80, 220, 80), -1)
        cv2.putText(image, label, (x1 + 4, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    banner = f"{len(preds)}  {infer_ms:.0f} ms"
    cv2.putText(image, banner, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, banner, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 1, cv2.LINE_AA)
    return image


def extend_for_prediction(camera: np.ndarray) -> np.ndarray:
    """Camera on the left, black continuation on the right in the same pixel space."""
    height, width = camera.shape[:2]
    canvas = np.zeros((height, width * 2, 3), dtype=np.uint8)
    canvas[:, :width] = camera
    cv2.line(canvas, (width, 0), (width, height - 1), (70, 70, 70), 1)
    cv2.putText(canvas, "out of sight", (width + 24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1, cv2.LINE_AA)
    return canvas


def encode(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise RuntimeError("Could not encode frame")
    return buf.tobytes()


def publish(image: np.ndarray, **status) -> None:
    global _jpeg
    payload = encode(image)
    with _lock:
        _jpeg = payload
        _status.update(status)


def placeholder(text: str) -> np.ndarray:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[:] = (18, 18, 18)
    cv2.putText(image, text, (40, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2, cv2.LINE_AA)
    return image


def open_capture(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source {source}")
    return cap


def load_model(api_key: str):
    from inference import get_model

    errors = []
    for model_id in (MODEL_ID, VERSION_ID):
        try:
            model = get_model(model_id=model_id, api_key=api_key)
            return model, model_id
        except Exception as exc:  # try the version id if the registry id is rejected
            errors.append(f"{model_id}: {exc}")
    raise RuntimeError(" ; ".join(errors))


def torch_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def ensure_yoloe_weights() -> str:
    """Use the local file, or download Ultralytics' yoloe-26l-seg.pt on first run."""
    if YOLOE_WEIGHTS.exists():
        return str(YOLOE_WEIGHTS)
    YOLOE_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLOE

    YOLOE("yoloe-26l-seg.pt")
    downloaded = Path("yoloe-26l-seg.pt")
    if downloaded.exists():
        downloaded.replace(YOLOE_WEIGHTS)
    if not YOLOE_WEIGHTS.exists():
        raise RuntimeError("Could not download yoloe-26l-seg.pt")
    return str(YOLOE_WEIGHTS)


def load_cutlery():
    """YOLOE-26l prompted with cutlery words. It is not limited to COCO tableware photos."""
    from ultralytics import YOLOE

    device = torch_device()
    model = YOLOE(ensure_yoloe_weights())
    model.set_classes(CUTLERY_PROMPTS, model.get_text_pe(CUTLERY_PROMPTS))

    def predict(frame: np.ndarray, confidence: float) -> list[dict]:
        result = model.predict(frame, conf=confidence, imgsz=640, verbose=False, device=device)[0]
        found = []
        if result.boxes is None:
            return found
        for box in result.boxes:
            name = result.names[int(box.cls[0])]
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            found.append({
                "x": (x1 + x2) / 2,
                "y": (y1 + y2) / 2,
                "width": x2 - x1,
                "height": y2 - y1,
                "confidence": float(box.conf[0]),
                "class": str(name),
            })
        return found

    return predict, "yoloe-26l-fork-knife-spoon-cutlery"


def run_video(source: str, model_id: str, predict, title: str, note: str) -> None:
    belt_tracker.reset()
    with _lock:
        _status["model_id"] = model_id
    publish(placeholder("Opening video…"), ready=False, message="Opening video", source=source, model_id=model_id)
    try:
        cap = open_capture(source)
    except Exception as exc:
        publish(placeholder(str(exc)), ready=False, message=str(exc), source=source, model_id=model_id)
        print(exc, flush=True)
        return

    loop = not source.isdigit()
    last_tick = time.perf_counter()
    frames = 0
    fps = 0.0
    print(f"model {model_id} ready, source {source}", flush=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            if not loop:
                publish(placeholder("Camera stopped producing frames."), ready=False, message="Camera ended", source=source, model_id=model_id)
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        captured = time.time() - belt_tracker.CAMERA_LATENCY_S
        with _lock:
            confidence = float(_status["confidence"])
        started = time.perf_counter()
        try:
            preds = predict(frame, confidence)
        except Exception as exc:
            publish(placeholder(str(exc)), ready=False, message=str(exc), source=source, model_id=model_id)
            print(f"predict failed: {exc}", flush=True)
            return
        infer_ms = (time.perf_counter() - started) * 1000
        frames += 1
        now = time.perf_counter()
        elapsed = now - last_tick
        if elapsed >= 0.5:
            fps = frames / elapsed
            frames = 0
            last_tick = now
        image = extend_for_prediction(draw(frame, preds, infer_ms))
        info = belt_tracker.annotate(image, captured, preds)
        speed = info["speed"]
        publish(
            image,
            ready=True,
            message="Live",
            detections=len(preds),
            infer_ms=round(infer_ms, 1),
            fps=round(fps, 1),
            source="camera" if source.isdigit() else source,
            model_id=model_id,
            title=title,
            note=note,
            belt_speed_mm_s=None if speed is None else round(speed, 1),
            tracks=sum(1 for track in info["tracks"] if track["trusted"]),
        )


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cutlery — live</title>
<style>
  :root { color-scheme: dark; font-family: system-ui, sans-serif; }
  body { margin: 0; background: #111; color: #eee; }
  header { display: flex; gap: 16px; align-items: center; padding: 12px 16px; background: #1c1c1c; border-bottom: 1px solid #333; }
  h1 { font-size: 16px; margin: 0; font-weight: 650; }
  main { padding: 16px; }
  img { width: 100%; background: #000; border-radius: 8px; display: block; }
  .row { display: flex; gap: 18px; align-items: center; flex-wrap: wrap; margin: 12px 0; }
  .muted { color: #999; font-size: 13px; }
  input[type=range] { width: 220px; }
  code { color: #d5d5d5; }
</style>
</head>
<body>
<header>
  <h1 id="title">Cutlery</h1>
  <span id="state" class="muted">Connecting…</span>
</header>
<main>
  <img id="view" alt="Live detections" src="/stream">
  <div class="row">
    <label>Confidence <output id="confLabel">0.40</output></label>
    <input id="conf" type="range" min="0.05" max="0.95" step="0.05" value="0.40">
    <span id="stats" class="muted"></span>
  </div>
  <p id="note" class="muted"></p>
  <p class="muted">Model <code id="model"></code></p>
</main>
<script>
const conf = document.querySelector("#conf");
const confLabel = document.querySelector("#confLabel");
let synced = false;
conf.addEventListener("input", () => {
  confLabel.textContent = Number(conf.value).toFixed(2);
  fetch("/confidence", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({confidence: Number(conf.value)})});
});
async function tick() {
  try {
    const data = await (await fetch("/stats")).json();
    if (!synced && typeof data.confidence === "number") {
      conf.value = data.confidence;
      confLabel.textContent = Number(data.confidence).toFixed(2);
      synced = true;
    }
    document.querySelector("#state").textContent = data.message;
    document.querySelector("#title").textContent = data.title || "Live";
    document.querySelector("#note").textContent = data.note || "";
    document.querySelector("#model").textContent = data.model_id || "";
    const speed = data.belt_speed_mm_s == null ? "belt speed learning" : `belt ${Math.round(data.belt_speed_mm_s)} mm/s`;
    document.querySelector("#stats").textContent = data.ready
      ? `${data.detections} detections · ${data.tracks || 0} tracked · ${speed} · ${data.infer_ms} ms · ${data.fps} fps · ${data.source}`
      : data.source || "";
  } catch (err) {
    document.querySelector("#state").textContent = "Reconnecting…";
  }
}
setInterval(tick, 500);
tick();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/stats":
            with _lock:
                body = __import__("json").dumps(_status).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    with _lock:
                        frame = _jpeg
                    if frame is None:
                        time.sleep(0.05)
                        continue
                    chunk = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/confidence":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            value = float(__import__("json").loads(raw).get("confidence", 0.4))
        except (ValueError, TypeError):
            self.send_error(400)
            return
        value = min(0.95, max(0.05, value))
        with _lock:
            _status["confidence"] = value
        body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description="Live local detector")
    parser.add_argument("--source", default="0", help="Webcam index, video file, or RTSP URL")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--detector", choices=("cutlery", "rfdetr"), default="cutlery")
    args = parser.parse_args()
    if args.detector == "cutlery":
        with _lock:
            _status["confidence"] = 0.10
        publish(placeholder("Loading cutlery model…"), ready=False, message="Loading YOLOE-26l", source=args.source, title="Cutlery")
        try:
            predict, model_id = load_cutlery()
        except Exception as exc:
            publish(placeholder("Model failed to load. See the terminal."), ready=False, message=str(exc), source=args.source)
            print(f"model load failed: {exc}", flush=True)
            predict, model_id = None, ""
        title = "Cutlery"
        note = "YOLOE-26l, prompted with fork, knife, spoon, and cutlery. Confidence starts at 0.10. The one prediction follows the knife that is moving, not a still object. When it leaves the camera, that point continues across the black screen for 3 seconds. Move it along the belt arrow."
    else:
        api_key = os.environ.get("ROBOFLOW_API_KEY", "")
        if not api_key:
            raise SystemExit(f"Set ROBOFLOW_API_KEY in {ROOT / '.env'}")
        publish(placeholder("Loading RF-DETR Nano weights…"), ready=False, message="Downloading model weights", source=args.source, title="Elongated metal object")
        try:
            model, model_id = load_model(api_key)
        except Exception as exc:
            publish(placeholder("Model failed to load. See the terminal."), ready=False, message=str(exc), source=args.source)
            print(f"model load failed: {exc}", flush=True)
            model, model_id = None, ""

        def predict(frame, confidence, model=model):
            if model is None:
                return []
            try:
                result = model.infer(frame, confidence=confidence)
            except TypeError:
                result = model.infer(frame)
            return predictions_of(result)

        title = "Elongated metal object"
        note = "Local RF-DETR Nano, trained on 13 frames of elongated metal. Raw detector, without the three-frame confirmation workflow."
    if predict is not None and model_id:
        threading.Thread(target=run_video, args=(args.source, model_id, predict, title, note), daemon=True).start()
    server = ThreadingHTTPServer((HOST, args.port), Handler)
    print(f"http://{HOST}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
