import cv2
import numpy as np

from eyetrax.calibration.common import (
    _pulse_and_capture,
    compute_grid_points_from_shape,
    wait_for_face_and_countdown,
)
from eyetrax.utils.screen import get_screen_size
from eyetrax.utils.video import open_camera


def run_dense_grid_calibration(
    gaze_estimator,
    *,
    rows: int = 5,
    cols: int = 5,
    margin_ratio: float = 0.10,
    order: str = "serpentine",
    pulse_d: float = 0.9,
    cd_d: float = 0.9,
    camera_index: int = 0,
    screen_index: int = 0,
    camera_rotate: int = 0,
) -> None:
    """
    Dense grid calibration for higher spatial precision.
    """
    sw, sh = get_screen_size(screen_index)

    cap = open_camera(camera_index)
    if not wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, 2, screen_index, camera_rotate=camera_rotate):
        cap.release()
        cv2.destroyAllWindows()
        return False

    pts = compute_grid_points_from_shape(
        rows, cols, sw, sh, margin_ratio=margin_ratio, order=order
    )
    res = _pulse_and_capture(gaze_estimator, cap, pts, sw, sh, pulse_d=pulse_d, cd_d=cd_d, camera_rotate=camera_rotate)
    cap.release()
    cv2.destroyAllWindows()

    if res is None:
        return False

    feats, targs = res
    if feats:
        gaze_estimator.train(np.array(feats), np.array(targs))
        print(f"[dense_grid] Calibrated with {len(feats)} samples from {rows}x{cols} grid")
    else:
        return False
    return True