import cv2
import numpy as np

from eyetrax.calibration.common import (
    _prepare_pulse_and_capture,
    _pulse_and_capture,
    compute_grid_points,
    wait_for_face_and_countdown,
)
from eyetrax.utils.screen import get_screen_size
from eyetrax.utils.video import open_camera


order = [(1, 1), (0, 0), (2, 0), (0, 2), (2, 2)]
pts = None

def prepare_5_point_calibration(screen_width, screen_height):
    pts = compute_grid_points(order, screen_width, screen_height)
    canvas = _prepare_pulse_and_capture(pts, screen_width, screen_height)
    return canvas


def run_5_point_calibration(gaze_estimator,
                            frame):
    """
    Faster five-point calibration
    """
    ok, canvas = _pulse_and_capture(gaze_estimator,
                                    frame)
    return ok, canvas
