"""Release regression checks. Run from the repo root; no camera/display needed."""

from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

CODE = Path(__file__).resolve().parents[1] / "LightTact_code"
sys.path.insert(0, str(CODE))
import calibrate
import lighttact
import run


class FakeCapture:
    def __init__(self, frames=(), opened=True, resolution=(960, 540)):
        self.frames, self.opened, self.released = iter(frames), opened, False
        self.properties = {
            cv2.CAP_PROP_FRAME_WIDTH: resolution[0], cv2.CAP_PROP_FRAME_HEIGHT: resolution[1],
            cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"MJPG"),
        }

    def isOpened(self):
        return self.opened

    def set(self, *_):
        return True

    def get(self, key):
        return self.properties[key]

    def getBackendName(self):
        return "V4L2"

    def read(self):
        frame = next(self.frames, None)
        return frame is not None, frame

    def release(self):
        self.released = True


class ReleaseChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = lighttact.load_config(CODE / "config.yaml")
        cls.saved = lighttact.Calibration.load(cls.cfg)
        cls.reference, cls.sample = lighttact.load_example_images(cls.cfg)

    def fixture(self, directory):
        directory = Path(directory)
        for name in ("config.yaml", "calibration.npz"):
            shutil.copy2(CODE / name, directory / name)
        return directory / "config.yaml"

    def gui(self, capture, keys):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(cv2, "VideoCapture", return_value=capture))
        for name in ("namedWindow", "imshow", "destroyWindow", "destroyAllWindows"):
            stack.enter_context(patch.object(cv2, name))
        stack.enter_context(patch.object(cv2, "getWindowProperty", return_value=-1))
        stack.enter_context(patch.object(cv2, "getWindowImageRect", return_value=(0, 0, 960, 540)))
        stack.enter_context(patch.object(cv2, "waitKey", side_effect=list(map(ord, keys))))
        return stack

    def test_calibration_replay_matches_example_exactly(self):
        actual = calibrate.build_calibration(self.reference, self.sample, self.cfg)
        for key in ("row_index", "col_index", "valid_mask"):
            np.testing.assert_array_equal(getattr(actual, key), getattr(self.saved, key))
        self.assertEqual(actual.metadata["mm_per_pixel_yx"], [14 / 444, 12.4 / 394])

    def test_processing_masks_invalid_region(self):
        images = run.process_frame(self.sample, self.reference, self.saved, self.cfg["contact"]["thresholds"])
        self.assertEqual(images["rectified"].shape, (445, 395, 3))
        self.assertEqual(set(np.unique(images["rectified_contact"])), {0, 255})
        self.assertFalse(images["rectified_contact"][~self.saved.valid_rect].any())

    def test_contact_thresholds_and_darkening(self):
        reference = np.full((1, 5, 3), 100, np.uint8)
        sample = np.array([[[0, 0, 0], [141, 100, 100], [131, 131, 100],
                            [121, 121, 121], [140, 100, 100]]], np.uint8)
        np.testing.assert_array_equal(lighttact.contact_mask(sample, reference, [40, 30, 20]),
                                      [[0, 255, 255, 255, 0]])
        self.assertFalse(lighttact.contact_mask(reference, reference, [40, 30, 20]).any())

    def test_blank_calibration_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expected 25 dots, found 0"):
            calibrate.build_calibration(self.reference, self.reference, self.cfg)

    def test_invalid_config_is_actionable(self):
        for text in ("", "camera: [", "camera: {}", "[]"):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                config = Path(directory) / "config.yaml"
                config.write_text(text)
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(run.main(["--config", str(config)]), 1)
                self.assertIn("Cannot run LightTact:", output.getvalue())

    def test_missing_calibration_is_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(directory)
            (Path(directory) / "calibration.npz").unlink()
            for main in (run.main, calibrate.main):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(main(["--config", str(config), "--offline"]), 1)
                self.assertIn("Run python calibrate.py", output.getvalue())

    def test_corrupt_bundles_are_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(directory)
            bundle = Path(directory) / "calibration.npz"
            for content in (b"not a numpy file", b"PK\x03\x04broken zip", b""):
                bundle.write_bytes(content)
                for main in (run.main, calibrate.main):
                    with self.subTest(content=content, main=main.__module__), redirect_stdout(io.StringIO()) as output:
                        self.assertEqual(main(["--config", str(config), "--offline"]), 1)
                        self.assertIn("Run python calibrate.py", output.getvalue())

    def test_invalid_arrays_and_metadata_are_actionable(self):
        with np.load(CODE / "calibration.npz", allow_pickle=False) as data:
            valid = {key: data[key] for key in data.files}
        variants = [dict(valid, metadata="{}"), dict(valid, metadata="not JSON"),
                    dict(valid, row_index=np.full_like(valid["row_index"], -1)),
                    dict(valid, row_index=valid["row_index"].astype(float)),
                    dict(valid, valid_mask=np.zeros_like(valid["valid_mask"])),
                    dict(valid, col_index=np.zeros((2, 2), np.int32)),
                    {key: value for key, value in valid.items() if key != "row_index"}]
        for key, value in [("format_version", 999), ("resolution", [0, 540]),
                           ("settings", {}), ("mm_per_pixel_yx", [float("nan"), 1])]:
            metadata = json.loads(str(valid["metadata"]))
            metadata[key] = value
            variants.append(dict(valid, metadata=json.dumps(metadata)))
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(directory)
            for i, arrays in enumerate(variants):
                with self.subTest(variant=i):
                    np.savez(Path(directory) / "calibration.npz", **arrays)
                    with redirect_stdout(io.StringIO()) as output:
                        self.assertEqual(run.main(["--config", str(config), "--offline"]), 1)
                    self.assertIn("Run python calibrate.py", output.getvalue())

    def test_changed_geometry_requires_recalibration(self):
        for section, key, value in [("camera", "resolution", [640, 480]), ("calibration", "roi_y", [254, 470])]:
            cfg = copy.deepcopy(self.cfg)
            cfg[section][key] = value
            with self.assertRaisesRegex(ValueError, "settings changed"):
                lighttact.Calibration.load(cfg)
        cfg = copy.deepcopy(self.cfg)
        cfg["contact"]["thresholds"] = [30, 20, 10]
        lighttact.Calibration.load(cfg)  # Changing runtime sensitivity does not invalidate maps.

    def test_failed_save_preserves_maps_and_images(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(directory)
            path = Path(directory) / "calibration.npz"
            before = path.read_bytes()
            for target in ("lighttact.np.savez_compressed", "lighttact.os.replace"):
                with patch(target, side_effect=OSError("simulated disk error")), self.assertRaises(OSError):
                    self.saved.save(path, self.reference, self.sample)
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["calibration.npz", config.name])

    def test_camera_failures_release_device(self):
        for capture in (FakeCapture(opened=False), FakeCapture(), FakeCapture(resolution=(640, 480))):
            with patch.object(cv2, "VideoCapture", return_value=capture), self.assertRaises(RuntimeError):
                with lighttact.Camera(self.cfg) as camera:
                    camera.read()
            self.assertTrue(capture.released)

    def test_camera_diagnostics_and_wrong_actual_frame(self):
        capture = FakeCapture([np.zeros((480, 640, 3), np.uint8)])
        with patch.object(cv2, "VideoCapture", return_value=capture) as constructor, \
             patch.object(sys, "platform", "linux"), redirect_stdout(io.StringIO()) as output:
            with self.assertRaisesRegex(ValueError, "got shape"):
                with lighttact.Camera(self.cfg) as camera:
                    camera.read()
        constructor.assert_called_once_with(4, cv2.CAP_V4L2)
        self.assertIn("960×540, 30 FPS, MJPG", output.getvalue())
        self.assertTrue(capture.released)

    def test_gtk_visibility_fallback_and_closed_windows(self):
        cases = [(1, None, False), (0, None, True), (-1, (0, 0, 960, 540), False),
                 (-1, (-1, -1, -1, -1), True)]
        for visibility, rect, expected in cases:
            with patch.object(cv2, "getWindowProperty", return_value=visibility), \
                 patch.object(cv2, "getWindowImageRect", return_value=rect):
                self.assertEqual(lighttact.window_closed("test"), expected)
        with patch.object(cv2, "getWindowProperty", side_effect=cv2.error("destroyed")):
            self.assertTrue(lighttact.window_closed("test"))

    def test_live_script_flow_with_mock_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(directory)
            capture = FakeCapture([self.reference] * 11 + [self.sample])
            with self.gui(capture, "yyy"), redirect_stdout(io.StringIO()):
                self.assertEqual(calibrate.main(["--config", str(config)]), 0)
            self.assertTrue(capture.released)
            capture = FakeCapture([self.reference] * 11 + [self.sample] * 2 + [self.reference] * 11 + [self.sample])
            with self.gui(capture, "ypryq"), redirect_stdout(io.StringIO()):
                self.assertEqual(run.main(["--config", str(config)]), 0)
            self.assertTrue(capture.released)
            self.assertEqual(len(list((Path(directory) / "snapshots").glob("*/*.png"))), 4)

    def test_cancel_keeps_existing_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(directory)
            bundle = Path(directory) / "calibration.npz"
            before = bundle.read_bytes()
            capture = FakeCapture([self.reference])
            with self.gui(capture, "q"), redirect_stdout(io.StringIO()):
                self.assertEqual(calibrate.main(["--config", str(config)]), 0)
            self.assertTrue(capture.released)
            self.assertEqual(bundle.read_bytes(), before)

    def test_both_clis_from_other_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(directory)
            for script in ("calibrate.py", "run.py"):
                result = subprocess.run([sys.executable, str(CODE / script), "--config", str(config), "--offline"],
                                        cwd=directory, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            saved = lighttact.Calibration.load(lighttact.load_config(config))
            np.testing.assert_array_equal(saved.row_index, self.saved.row_index)
            self.assertEqual(len(list((Path(directory) / "snapshots").glob("*/stats.csv"))), 1)


if __name__ == "__main__":
    unittest.main()
