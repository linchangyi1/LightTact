# LightTact

Calibrate your sensor, then run live contact sensing. Tested with Python 3.10,
a connected camera, and a desktop display.

## 1. Setup

```bash
cd LightTact_code
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Set `camera.device` in `config.yaml` to your camera index or device path.
On Ubuntu, install `python3-venv` and `v4l-utils` if those commands are missing.
Find the camera and set manual exposure (verified for the DECXIN5M sensor):

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video4 --set-ctrl=auto_exposure=1,exposure_time_absolute=200
```

Replace `/dev/video4` with your device. If a control name is unrecognized, check
`v4l2-ctl -d /dev/video4 --list-ctrls`. Keep exposure and illumination fixed.
Startup prints the actual device, backend, resolution, frame rate, and format.

## 2. Calibrate

```bash
python calibrate.py
```

1. Leave the sensor untouched and press **y** to capture a reference.
2. Hold the calibration board against the sensor and press **y**.
3. Check the preview, then press **y** to save or **q** to cancel.

The default board is a **5×5 dot grid with 3 mm spacing**. Adjust the camera
resolution, ROI, and board settings in `config.yaml` to match your sensor.
Calibration and example images are saved together in `calibration.npz`.
The included file is an example; calibrate your own sensor before use.

## 3. Run

```bash
python run.py
```

Leave the sensor untouched and press **y**. The windows show the rectified
image and contact mask. Press **p** to save a snapshot, **r** to reset the
reference, or **q** to quit.

Tested on Ubuntu 22.04 / Python 3.10 with a DECXIN5M camera at 960×540,
30 FPS, MJPG. Dependency versions are pinned in `requirements.txt`.
Offline calibration replay and live runtime were verified; a fresh physical
calibration still needs validation with the dot board.

Developer checks (from the repository root): `python -m unittest discover -s tests -v`.

Code license: [BSD-3-Clause](LICENSE).

Without a sensor, try the included example with `python run.py --offline`.
