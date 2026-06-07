import cv2
import numpy as np

from eyetrax.calibration.common import wait_for_face_and_countdown
from eyetrax.utils.screen import get_screen_size
from eyetrax.utils.video import open_camera


def run_lissajous_calibration(gaze_estimator, camera_index: int = 0, screen_index: int = 0, camera_rotate: int = 0):
    """
    Moves a calibration point along a Lissajous curve
    """
    sw, sh = get_screen_size(screen_index)

    cap = open_camera(camera_index)
    if not wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, 2, screen_index, camera_rotate=camera_rotate):
        cap.release()
        cv2.destroyAllWindows()
        return False

    A, B, a, b, d = sw * 0.4, sh * 0.4, 3, 2, 0

    def curve(t):
        return (A * np.sin(a * t + d) + sw / 2, B * np.sin(b * t) + sh / 2)

    total_time = 5.0
    fps = 60
    frames = int(total_time * fps)
    feats, targs = [], []
    acc = 0

    for i in range(frames):
        frac = i / (frames - 1)
        spd = 0.3 + 0.7 * np.sin(np.pi * frac)
        acc += spd / fps
    end = acc if acc >= 1e-6 else 1e-6
    acc = 0

    for i in range(frames):
        frac = i / (frames - 1)
        spd = 0.3 + 0.7 * np.sin(np.pi * frac)
        acc += spd / fps
        t = (acc / end) * (2 * np.pi)
        ret, f = cap.read()
        if not ret:
            continue
        if camera_rotate == 1:
            f = cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE)
        elif camera_rotate == 2:
            f = cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imshow("thumbnail", f)
        x, y = curve(t)
        c = np.zeros((sh, sw, 3), dtype=np.uint8)
        cv2.circle(c, (int(x), int(y)), 20, (0, 255, 0), -1)
        cv2.imshow("Calibration", c)
        if cv2.waitKey(1) == 27:
            return False
        ft, blink = gaze_estimator.extract_features(f)
        if ft is not None and not blink:
            feats.append(ft)
            targs.append([x, y])

    cap.release()
    cv2.destroyAllWindows()
    if feats:
        gaze_estimator.train(np.array(feats), np.array(targs))
    else:
        return False
    return True
