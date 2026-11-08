"""Shared camera, contact detection, and calibration I/O for the two scripts."""

import json
import os
from pathlib import Path
import sys
import tempfile
from zipfile import BadZipFile
import cv2
import numpy as np
import yaml


DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def load_config(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    try:
        camera, calibration = cfg["camera"], cfg["calibration"]
        width, height = camera["resolution"]
        y0, y1 = calibration["roi_y"]
        rows, cols = calibration["grid"]
        integers = [width, height, y0, y1, rows, cols, camera["reference_frames"]]
        if any(type(value) is not int for value in integers):
            raise ValueError("Resolution, ROI, grid, and reference_frames must be integers.")
        if width <= 0 or height <= 0 or not 0 <= y0 < y1 <= height:
            raise ValueError("The ROI must fit inside the camera resolution.")
        if rows < 3 or cols < 3 or rows % 2 == 0 or cols % 2 == 0:
            raise ValueError("Use an odd grid with at least 3 rows and 3 columns.")
        for key, positive in [("spacing_mm", True), ("margin_mm", False)]:
            values = calibration[key]
            if len(values) != 2 or not np.isfinite(values).all():
                raise ValueError(f"{key} must contain two finite numbers.")
            if any(v <= 0 if positive else v < 0 for v in values):
                raise ValueError(f"Invalid {key}.")
        for thresholds in [calibration["thresholds"], cfg["contact"]["thresholds"]]:
            if len(thresholds) != 3 or not np.isfinite(thresholds).all():
                raise ValueError("Thresholds must contain three finite numbers.")
            if not 255 >= thresholds[0] >= thresholds[1] >= thresholds[2] >= 0:
                raise ValueError("Thresholds must be descending values between 0 and 255.")
        if not np.isfinite(camera["fps"]) or camera["fps"] <= 0 or camera["reference_frames"] < 1:
            raise ValueError("FPS and reference_frames must be positive.")
        if not isinstance(camera["device"], (int, str)):
            raise ValueError("Camera device must be an index or device path.")
        cfg["calibration_file"] = (path.parent / cfg["calibration_file"]).resolve()
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid configuration in {path}: {exc}") from exc
    cfg["base_dir"] = path.parent
    return cfg


def check_frame(image, resolution):
    width, height = resolution
    if image is None or image.shape != (height, width, 3) or image.dtype != np.uint8:
        actual = "no image" if image is None else f"shape {image.shape}, dtype {image.dtype}"
        raise ValueError(f"Expected a {width}×{height} uint8 color frame; got {actual}. "
                         "Check camera resolution; do not resize frames to fit an old calibration.")


def contact_mask(image, reference, thresholds):
    if image.shape != reference.shape:
        raise ValueError("Reference and camera image must have the same shape.")
    difference = image.astype(np.int16) - reference.astype(np.int16)
    one, two, three = thresholds
    contact = (np.any(difference > one, axis=2)
               | ((difference > two).sum(axis=2) >= 2)
               | np.all(difference > three, axis=2))
    return contact.astype(np.uint8) * 255


def window_closed(name):
    """Older GTK backends do not implement WND_PROP_VISIBLE (return -1)."""
    try:
        visible = cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE)
        if visible >= 0:
            return visible == 0
        # GTK returns negative dimensions once the window has been destroyed.
        _, _, width, height = cv2.getWindowImageRect(name)
        return width < 0 or height < 0
    except cv2.error:
        return True  # Some backends raise when the window has been destroyed.


class Camera:
    def __init__(self, cfg):
        settings = cfg["camera"]
        self.resolution = settings["resolution"]
        self.reference_frames = settings["reference_frames"]
        # Use the Linux camera driver directly instead of an auto-selected GStreamer pipeline.
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(settings["device"], backend)
        try:
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera {settings['device']!r}; check config.yaml.")
            if backend == cv2.CAP_V4L2:
                # The sensor supports 960×540 at 30 FPS in MJPG (YUYV is limited to 15).
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, settings["fps"])
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join(chr((fourcc >> (8 * i)) & 255) for i in range(4)).strip("\x00")
            print(f"Camera {settings['device']!r}: {self.cap.getBackendName()}, "
                  f"{width}×{height}, {fps:g} FPS, {codec or 'unknown format'}")
            if width > 0 and height > 0 and [width, height] != self.resolution:
                raise RuntimeError(f"Camera negotiated {width}×{height}, but calibration requires "
                                   f"{self.resolution[0]}×{self.resolution[1]}. "
                                   "Select a supported camera mode or recalibrate.")
            if fps > 0 and abs(fps - settings["fps"]) > 0.5:
                print(f"Note: requested {settings['fps']:g} FPS; camera reports {fps:g} FPS.")
        except Exception:
            self.cap.release()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cap.release()

    def read(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Camera frame capture failed; check the connection and camera settings.")
        check_frame(frame, self.resolution)
        return frame

    def capture(self, instruction, average=False):
        print(f"{instruction} Press y to capture, q to quit.")
        window = "Capture — y: capture, q: quit"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        try:
            while True:
                frame = self.read()
                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or window_closed(window):
                    raise KeyboardInterrupt
                if key == ord("y"):
                    break
            if not average:
                return frame
            total = np.zeros_like(frame, dtype=np.float64)
            for _ in range(self.reference_frames):
                total += self.read()
            return (total / self.reference_frames).astype(np.uint8)
        finally:
            if not window_closed(window):
                cv2.destroyWindow(window)


class Calibration:
    def __init__(self, row_index, col_index, valid_mask, metadata):
        self.row_index = row_index
        self.col_index = col_index
        self.valid_mask = valid_mask
        self.metadata = metadata
        try:
            if type(metadata["format_version"]) is not int or metadata["format_version"] != 1:
                raise ValueError("Unsupported calibration format.")
            width, height = metadata["resolution"]
            y0, y1 = metadata["settings"]["roi_y"]
            if any(type(v) is not int for v in (width, height, y0, y1)) or min(width, height) <= 0:
                raise ValueError("Invalid calibration resolution or ROI.")
            scale = metadata["mm_per_pixel_yx"]
            origin = metadata["origin_mm_yx"]
            if (len(scale) != 2 or len(origin) != 2 or not np.isfinite(scale + origin).all()
                    or min(scale) <= 0):
                raise ValueError("Invalid calibration physical scale.")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid calibration metadata: {exc}") from exc
        if not 0 <= y0 < y1 <= height or valid_mask.shape != (y1 - y0, width):
            raise ValueError("Calibration ROI and valid mask do not match.")
        if valid_mask.dtype != np.uint8 or not np.isin(valid_mask, [0, 255]).all() or not valid_mask.any():
            raise ValueError("Calibration valid mask must contain a nonempty binary region.")
        if row_index.ndim != 2 or row_index.size == 0 or row_index.shape != col_index.shape:
            raise ValueError("Invalid calibration map dimensions.")
        for indices, bound in [(row_index, y1 - y0), (col_index, width)]:
            if not np.issubdtype(indices.dtype, np.integer) or indices.min() < 0 or indices.max() >= bound:
                raise ValueError("Calibration map contains invalid image coordinates.")
        self.valid_rect = valid_mask[row_index, col_index] != 0

    def rectify(self, image, masked=True):
        width, height = self.metadata["resolution"]
        if image.shape[:2] != (height, width):
            raise ValueError("Image resolution differs from calibration; recalibrate at this resolution.")
        y0, y1 = self.metadata["settings"]["roi_y"]
        result = image[y0:y1][self.row_index, self.col_index]
        if masked:
            result[~self.valid_rect] = 0
        return result

    def save(self, path, reference, sample):
        check_frame(reference, self.metadata["resolution"])
        check_frame(sample, self.metadata["resolution"])
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Replace maps and captured images together only after a successful write.
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            np.savez_compressed(temporary, row_index=self.row_index, col_index=self.col_index,
                                valid_mask=self.valid_mask, metadata=json.dumps(self.metadata),
                                reference=reference, sample=sample)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, cfg):
        path = cfg["calibration_file"]
        data = read_calibration_arrays(path, ("row_index", "col_index", "valid_mask", "metadata"))
        try:
            result = cls(data["row_index"], data["col_index"], data["valid_mask"],
                         json.loads(str(data["metadata"])))
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
            raise ValueError(f"Invalid calibration at {path}: {exc}. Run python calibrate.py again.") from exc
        if (result.metadata["resolution"] != cfg["camera"]["resolution"]
                or result.metadata["settings"] != cfg["calibration"]):
            raise ValueError("Calibration settings changed. Run python calibrate.py again.")
        return result


def read_calibration_arrays(path, names):
    """Read only numeric/string arrays, reporting incomplete or corrupt bundles clearly."""
    if not path.is_file():
        raise FileNotFoundError(f"No calibration at {path}. Run python calibrate.py first.")
    try:
        with path.open("rb") as stream, np.load(stream, allow_pickle=False) as data:
            missing = set(names) - set(data.files)
            if missing:
                raise ValueError(f"missing {', '.join(sorted(missing))}")
            return {name: data[name] for name in names}
    except (OSError, ValueError, TypeError, AttributeError, KeyError, EOFError, BadZipFile) as exc:
        raise ValueError(f"Cannot read calibration at {path}: {exc}. Run python calibrate.py again.") from exc


def load_example_images(cfg):
    data = read_calibration_arrays(cfg["calibration_file"], ("reference", "sample"))
    images = [data["reference"], data["sample"]]
    for image in images:
        check_frame(image, cfg["camera"]["resolution"])
    return images


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save image to {path}.")
