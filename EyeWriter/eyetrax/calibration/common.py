import time

import cv2
import numpy as np
import screeninfo
from enum import StrEnum, auto


def compute_grid_points(order, sw: int, sh: int, margin_ratio: float = 0.10):
    """
    Translate grid (row, col) indices into absolute pixel locations
    """
    if not order:
        return []

    max_r = max(r for r, _ in order)
    max_c = max(c for _, c in order)

    mx, my = int(sw * margin_ratio), int(sh * margin_ratio)
    gw, gh = sw - 2 * mx, sh - 2 * my

    step_x = 0 if max_c == 0 else gw / max_c
    step_y = 0 if max_r == 0 else gh / max_r

    return [(mx + int(c * step_x), my + int(r * step_y)) for r, c in order]


def compute_grid_points_from_shape(
    rows: int,
    cols: int,
    sw: int,
    sh: int,
    margin_ratio: float = 0.10,
    order: str = "default",
) -> list[tuple[int, int]]:
    """
    Generate (x, y) pixel coordinates for a rows x cols grid.

    `order` controls traversal:
    - "default": row-major
    - "serpentine": snake pattern (reduces large jumps between rows)
    """
    if rows <= 0 or cols <= 0:
        raise ValueError(f"rows and cols must be > 0 (got rows={rows}, cols={cols})")
    if not 0.0 <= margin_ratio < 0.5:
        raise ValueError(f"margin_ratio must be in [0.0, 0.5) (got {margin_ratio})")

    if order == "default":
        indices = [(r, c) for r in range(rows) for c in range(cols)]
    elif order == "serpentine":
        indices = []
        for r in range(rows):
            cols_range = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
            indices.extend((r, c) for c in cols_range)
    else:
        raise ValueError(f"unknown order '{order}' (expected 'default' or 'serpentine')")

    return compute_grid_points(indices, sw, sh, margin_ratio)


class STATE(StrEnum):
    PREPARING = auto()
    PULSING = auto()
    CAPTURING = auto()

_state = None
_pts = None
_pts_idx = None
_ps = None
_pulse_d = None
_cs = None
_cd_d = None
_countdown_start = None
_countdown = False
_background = None
_final_radius = None
_feats, _targs = None, None


def wait_for_face_and_countdown(gaze_estimator, frame, dur: int = 2):
    """
    Waits for a face to be detected (not blinking), then shows a countdown ellipse
    """
    global _countdown_start, _countdown

    canvas = _background.copy()
    sw = canvas.shape[1]
    sh = canvas.shape[0]

    f, blink = gaze_estimator.extract_features(frame)
    face = f is not None and not blink
    now = time.time()
    if face:
        if not _countdown:
            _countdown_start = now
            _countdown = True
        elapsed = now - _countdown_start
        if elapsed >= dur:
            return True, canvas
        t = elapsed / dur
        e = t * t * (3 - 2 * t)
        ang = 360 * (1 - e)
        cv2.ellipse(
            canvas,
            (sw // 2, sh // 2),
            (50, 50),
            0,
            -90,
            -90 + ang,
            (0, 255, 0),
            -1,
        )
    else:
        _countdown = False
        _countdown_start = None
        txt = "Face not detected"
        fs = 2
        thick = 3
        size, _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
        tx = (sw - size[0]) // 2
        ty = (sh + size[1]) // 2
        cv2.putText(
            canvas, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), thick
        )
    return False, canvas

def _pulse(x, y):
    e = time.time() - _ps
    if e > _pulse_d:
        return True, None

    canvas = _background.copy()
    radius = 15 + int(15 * abs(np.sin(2 * np.pi * e)))
    _final_radius = radius
    cv2.circle(canvas, (x, y), radius, (0, 255, 0), -1)

    return False, canvas

def _capture(x, y):
    e = time.time() - _cs
    if e > _cd_d:
        return True, None

    canvas = _background.copy()
    cv2.circle(canvas, (x, y), _final_radius, (0, 255, 0), -1)
    t = e / _cd_d
    ease = t * t * (3 - 2 * t)
    ang = 360 * (1 - ease)
    cv2.ellipse(canvas, (x, y), (40, 40), 0, -90, -90 + ang, (255, 255, 255), 4)

    return False, canvas

def _pulse_and_capture(gaze_estimator,
                       frame):
    """
    Shared pulse-and-capture loop for each calibration point
    """
    global _state, _ps, _cs, _pts_idx

    x, y = _pts[_pts_idx]

    if _state == STATE.PREPARING:
        ok, canvas = wait_for_face_and_countdown(gaze_estimator, frame)
        if ok:
            _ps = time.time()
            _state = STATE.PULSING
            ok, canvas = _pulse(x, y)
    elif _state == STATE.PULSING:
        ok, canvas = _pulse(x, y)
        if ok:
            _cs = time.time()
            _state = STATE.CAPTURING
            ok, canvas = _capture(x, y)
    elif _state == STATE.CAPTURING:
        ok, canvas = _capture(x, y)
        print("Extracting features")
        ft, blink = gaze_estimator.extract_features(frame)
        if ft is not None and not blink:
            print("Point added")
            _feats.append(ft)
            _targs.append([x, y])
        if ok:
            _pts_idx += 1
            _ps = time.time()
            _state = STATE.PULSING
            if _pts_idx == len(_pts):
                print(f"Training with {len(_feats)} points")
                gaze_estimator.train(np.array(_feats), np.array(_targs))
                return True, _background
            else:
                x, y = _pts[_pts_idx]
                ok, canvas = _pulse(x, y)

    # capture

    return False, canvas

def _prepare_pulse_and_capture(pts,
                               screen_width: int,
                               screen_height: int,
                               pulse_d: float = 1.0,
                               cd_d: float = 1.0):
    global _state, _pts, _pts_idx, _ps, _pulse_d, _cd_d, _background, _final_radius, _feats, _targs, _countdown
    _state = STATE.PREPARING
    _pts = pts
    _pts_idx = 0
    _pulse_d = pulse_d
    _cd_d = cd_d
    _background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    _final_radius = 20
    _feats, _targs = [], []
    _countdown = False
    _ps = time.time()
    x, y = _pts[0]
    _, canvas = _pulse(x, y)
    return canvas
