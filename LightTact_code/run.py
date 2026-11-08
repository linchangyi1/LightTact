"""Show rectified LightTact images and contact masks using saved calibration."""

import argparse
import csv
from datetime import datetime
import cv2
import numpy as np
import yaml

from lighttact import (DEFAULT_CONFIG, Calibration, Camera, contact_mask,
                       load_config, load_example_images, window_closed, write_image)


def process_frame(image, reference, calibration, thresholds):
    contact = contact_mask(image, reference, thresholds)
    return {"raw": image, "contact": contact,
            "rectified": calibration.rectify(image),
            "rectified_contact": calibration.rectify(contact)}


def save_snapshot(images, root):
    directory = root / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    directory.mkdir(parents=True)
    with (directory / "stats.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "width", "height", "gray_mean", "gray_std", "contact_fraction"])
        for name, image in images.items():
            write_image(directory / f"{name}.png", image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            fraction = float(np.count_nonzero(image) / image.size) if image.ndim == 2 else ""
            writer.writerow([name, image.shape[1], image.shape[0],
                             float(gray.mean()), float(gray.std()), fraction])
    print(f"Saved images and statistics to {directory}")
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--offline", action="store_true",
                        help="Process saved reference/sample images and save a snapshot without a camera or GUI.")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        calibration = Calibration.load(cfg)
        thresholds = cfg["contact"]["thresholds"]
        snapshots = cfg["base_dir"] / "snapshots"
        if args.offline:
            reference, sample = load_example_images(cfg)
            save_snapshot(process_frame(sample, reference, calibration, thresholds), snapshots)
            return 0
        try:
            with Camera(cfg) as camera:
                reference = camera.capture("Leave the sensor untouched to capture the reference.", average=True)
                windows = {"rectified": "LightTact image", "rectified_contact": "LightTact contact"}
                for window in windows.values():
                    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                print("Press p to save a snapshot, r to recapture the reference, q to quit.")
                while True:
                    images = process_frame(camera.read(), reference, calibration, thresholds)
                    for name, window in windows.items():
                        cv2.imshow(window, images[name])
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27) or any(window_closed(w) for w in windows.values()):
                        break
                    if key == ord("p"):
                        save_snapshot(images, snapshots)
                    elif key == ord("r"):
                        reference = camera.capture("Leave the sensor untouched.", average=True)
        finally:
            cv2.destroyAllWindows()
        return 0
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, RuntimeError, cv2.error, yaml.YAMLError) as exc:
        print(f"Cannot run LightTact: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
