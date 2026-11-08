"""Capture a reference and dot grid, then save calibration for run.py."""

import argparse

import cv2
import numpy as np
from scipy.interpolate import Rbf
import yaml

from lighttact import (DEFAULT_CONFIG, Calibration, Camera, check_frame,
                       contact_mask, load_config, load_example_images, window_closed)


def find_grid(mask, rows, cols):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for contour in contours:
        moments = cv2.moments(contour)
        if cv2.contourArea(contour) >= 1.0 and moments["m00"] != 0:
            points.append([moments["m10"] / moments["m00"],
                           moments["m01"] / moments["m00"]])
    if len(points) != rows * cols:
        raise ValueError(f"Expected {rows * cols} dots, found {len(points)}. "
                         "Check board contact, ROI, exposure, and calibration thresholds.")
    points = np.array(points, dtype=np.float32)
    points = points[np.lexsort((points[:, 0], points[:, 1]))]
    bands = np.array_split(points, rows)
    points = np.vstack([band[np.argsort(band[:, 0])] for band in bands])
    return points[:, ::-1].copy()  # row-major (y, x) in cropped raw space


def build_calibration(reference, sample, cfg):
    """Fit the original cubic RBF mapping, without opening a camera or writing files."""
    check_frame(reference, cfg["camera"]["resolution"])
    check_frame(sample, cfg["camera"]["resolution"])
    settings = cfg["calibration"]
    y0, y1 = settings["roi_y"]
    rows, cols = settings["grid"]
    row_mm, col_mm = settings["spacing_mm"]
    vertical_mm, horizontal_mm = settings["margin_mm"]
    contact = contact_mask(sample, reference, settings["thresholds"])[y0:y1]
    points = find_grid(contact, rows, cols)
    ul, ur, bl, br = points[[0, cols - 1, (rows - 1) * cols, rows * cols - 1]]
    top = float(np.linalg.norm(ur - ul))
    bottom = float(np.linalg.norm(br - bl))
    left = float(np.linalg.norm(bl - ul))
    right = float(np.linalg.norm(br - ur))
    col_pixels = (top + bottom) / (2.0 * (cols - 1))
    row_pixels = (left + right) / (2.0 * (rows - 1))
    pixels_per_mm = 0.5 * (row_pixels / row_mm + col_pixels / col_mm)

    physical = np.zeros_like(points, dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            physical[row * cols + col] = [(row - rows // 2) * row_mm,
                                          (col - cols // 2) * col_mm]
    row_fit = Rbf(physical[:, 0], physical[:, 1], points[:, 0], function="cubic")
    col_fit = Rbf(physical[:, 0], physical[:, 1], points[:, 1], function="cubic")
    height_mm = (rows - 1) * row_mm + 2.0 * vertical_mm
    width_mm = (cols - 1) * col_mm + 2.0 * horizontal_mm
    height = max(8, int(round(height_mm * pixels_per_mm)))
    width = max(8, int(round(width_mm * pixels_per_mm)))
    ys = np.linspace(-height_mm / 2.0, height_mm / 2.0, height, dtype=np.float32)
    xs = np.linspace(-width_mm / 2.0, width_mm / 2.0, width, dtype=np.float32)
    col_mesh, row_mesh = np.meshgrid(xs, ys)
    row_index = np.clip(row_fit(row_mesh, col_mesh), 0, contact.shape[0] - 1).astype(np.int32)
    col_index = np.clip(col_fit(row_mesh, col_mesh), 0, contact.shape[1] - 1).astype(np.int32)
    hull = cv2.convexHull(points[:, ::-1].reshape(-1, 1, 2).astype(np.float32))
    valid_mask = np.zeros(contact.shape, dtype=np.uint8)
    cv2.fillConvexPoly(valid_mask, hull.astype(np.int32), 255)
    metadata = {
        "format_version": 1,
        "resolution": cfg["camera"]["resolution"],
        "settings": settings,
        # linspace includes both endpoints, so use N-1 for the actual output scale.
        "mm_per_pixel_yx": [height_mm / (height - 1), width_mm / (width - 1)],
        "origin_mm_yx": [-height_mm / 2.0, -width_mm / 2.0],
    }
    return Calibration(row_index, col_index, valid_mask, metadata)


def save_calibration(calibration, reference, sample, cfg):
    path = cfg["calibration_file"]
    calibration.save(path, reference, sample)
    height, width = calibration.row_index.shape
    print(f"Saved calibration to {path} ({width}×{height} rectified pixels).")
    print("Next: python run.py")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--offline", action="store_true",
                        help="Rebuild from the images in calibration.npz without a camera or GUI.")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        if args.offline:
            reference, sample = load_example_images(cfg)
            calibration = build_calibration(reference, sample, cfg)
        else:
            try:
                with Camera(cfg) as camera:
                    reference = camera.capture("Leave the sensor untouched.", average=True)
                    sample = camera.capture("Hold the calibration dot grid against the sensor.")
                    calibration = build_calibration(reference, sample, cfg)
                    window = "Calibration preview — y: save, q: cancel"
                    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                    cv2.imshow(window, calibration.rectify(sample))
                    print("Check that the dots form a regular grid. Press y to save, q to cancel.")
                    while True:
                        key = cv2.waitKey(50) & 0xFF
                        if key == ord("y"):
                            break
                        if key in (ord("q"), 27) or window_closed(window):
                            raise KeyboardInterrupt
            finally:
                cv2.destroyAllWindows()
        save_calibration(calibration, reference, sample, cfg)
        return 0
    except KeyboardInterrupt:
        print("Calibration cancelled; existing calibration kept.")
        return 0
    except (OSError, ValueError, RuntimeError, cv2.error, yaml.YAMLError) as exc:
        print(f"Calibration failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
