# Robot tracker test

Live cutlery detector plus a belt-motion prediction. Built to try on a camera before using it on a sorting line.

The page is `http://127.0.0.1:8770`. The camera image is on the left. A black panel is on the right.

## What you should see

- Green boxes are detections from YOLOE-26l, prompted with the words `fork`, `knife`, `spoon`, and `cutlery`. This is an open-vocabulary model, not a COCO detector. A COCO model was tried and missed the cutlery.
- Confidence starts at **0.10**. Raise the slider if it marks things that are not cutlery.
- There is **one** orange prediction. It follows the detection that is **moving**. A still object, even if it is also called a knife, does not get the prediction.
- While the knife stays in view, every new frame is added to the fit, so the speed and position keep tightening.
- Move the knife along the white **belt** arrow, off the right side of the camera. After it leaves the image, the same point continues across the black panel for **3 seconds**, then disappears.
- The millimetre speed on screen uses placeholder marker points. It is not a measured belt speed until you calibrate.

## Run

Python 3.11 or newer. The first start downloads `yoloe-26l-seg.pt` (about 76 MB) into `models/`, then a text encoder. Later starts use those files. Frames stay on this machine.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python live.py --source 0
```

Open `http://127.0.0.1:8770`. On macOS, allow the terminal to use the camera if prompted.

Stop it with Ctrl-C.

Other sources:

```bash
python live.py --source /path/to/video.mp4
python live.py --source "rtsp://user:pass@host:554/stream"
python live.py --port 8770
```

`--source 0` is the default webcam.

## Files

| File | Role |
| --- | --- |
| `live.py` | Camera loop, YOLOE, and the browser page |
| `belt_tracker.py` | Pixel-to-millimetre homography, the one track, speed fit, and the 3 second coast |
| `test_belt_tracker.py` | Simulated belt. Speed within 0.3 mm/s and pick time within 5 ms on that simulation. Also checks that a moving object is preferred over a still one. |

```bash
python -m unittest test_belt_tracker.py
```

## Before the millimetres mean anything

`belt_tracker.py` converts camera pixels to belt millimetres with four markers. The values in `PIXEL_PTS` and `BELT_MM` are placeholders. `x` in `BELT_MM` points in the direction of travel. The white arrow on the page is that direction in the image (roughly to the right with the placeholder points).

Replace, in `belt_tracker.py`:

- `PIXEL_PTS` — pixel positions of the markers in one camera frame
- `BELT_MM` — where those markers sit on the belt, in millimetres
- `PICK_LINE_MM` — belt x where a robot would pick (placeholder `900`)
- `CAMERA_LATENCY_S` — fixed delay from the sensor to the timestamp you pass in. Left at `0`. At 250 mm/s, 40 ms of uncounted delay is 10 mm of position error.

`pick_plan()` returns, for the trusted object still upstream of the pick line, `t_pick` and `y_mm`. `predict_position(track_id, t_query)` returns belt `(x_mm, y_mm)` at an absolute time. The live page draws that position at the current time, in the same pixel coordinates as the camera, so it slides onto the black panel when it passes the right edge.

The robot and this computer need the same clock if you send an absolute pick time. Otherwise send a delay from now. A belt encoder is more reliable than the camera speed if the belt changes speed. This test does not read an encoder.

## Optional private model

`--detector rfdetr` loads Roboflow model `easysort/elongated-metal-object-trial-1-rfdetr-nano-t1` locally through the `inference` package. That model was trained on 13 frames of “elongated metal object” and is not the one that worked for cutlery in this test. Use the default instead.

```bash
pip install inference
cp .env.example .env
# put the Roboflow API key in .env
python live.py --detector rfdetr --source 0
```

Do not commit `.env`.

## Agent notes

- Run `python live.py --source 0` and open port 8770. Do not start `belt_tracker.py` directly if you want the page. That file’s `__main__` only prints a pick plan for the RF-DETR pipeline.
- Do not swap back to a COCO YOLO for cutlery. It did not see the knife.
- One prediction per camera. It switches to an object that is moving at least `MOVE_MM_S` (60 mm/s) and will not let a still detection steal it. After the moving object disappears, the prediction coasts for `FORGET_AFTER_S` (3 s).
- The black panel is the camera image extended to the right in the same pixel coordinates. The placeholder homography maps belt +x to image +x, about 550 px per 300 mm. If the knife leaves through a different edge, the point will not cross the black panel.
- GPU: CUDA, then Apple MPS, then CPU.
- Tests do not need a camera or weights.
