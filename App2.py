import sys

import os
import time
from math import atan2, pi

from enum import StrEnum, auto

import cv2
import numpy as np

from eyetrax.calibration import (
    run_5_point_calibration,
    run_9_point_calibration,
    run_dense_grid_calibration,
    run_lissajous_calibration,
)
from eyetrax.cli import parse_common_args
from eyetrax.filters import (
    KDESmoother,
    KalmanEMASmoother,
    KalmanSmoother,
    NoSmoother,
    make_kalman,
)
from eyetrax.gaze import GazeEstimator
from eyetrax.utils.draw import draw_cursor, make_thumbnail
from eyetrax.utils.screen import get_screen_size
from eyetrax.utils.video import camera, fullscreen, iter_frames

# Audio playing from https://stackoverflow.com/questions/76696178/how-to-make-play-a-sound-from-a-string
from pydub.playback import play
from pydub import AudioSegment
from gtts import gTTS
from io import BytesIO

def print_help():
    print("Demo arguments")
    print("--filter [kalman | kalman_ema | kde]")
    print("--camera camera_index")
    print("--calibration [9p | 5p | dense | lissajous]")
    print("--background background_path")
    print("--confidence confidence_level")
    print("--ema_alpha ema_alpha")
    exit(0)

if len(sys.argv) > 1:
    if sys.argv[1] == '--help':
        print_help()

class DIRECTION(StrEnum):
    CENTER = auto()
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()
    NORTHEAST = auto()
    NORTHWEST = auto()
    SOUTHEAST = auto()
    SOUTHWEST = auto()
    BLINK = auto()
    NO_FACE = auto()
    NONE = auto()

COLORS = {
    DIRECTION.CENTER: (0,0,0),
    DIRECTION.UP: (0,0,150),
    DIRECTION.DOWN: (0,0,150),
    DIRECTION.LEFT: (0,150,0),
    DIRECTION.RIGHT: (0,150,0),
    DIRECTION.NORTHEAST: (150,0,0),
    DIRECTION.NORTHWEST: (150,0,0),
    DIRECTION.SOUTHEAST: (150,0,0),
    DIRECTION.SOUTHWEST: (150,0,0),
    DIRECTION.BLINK: (255,255,255),
    DIRECTION.NO_FACE: (150,0,150),
    DIRECTION.NONE: (150,150,150)
}

class Letter:
    def __init__(self, char, parent=None, x=0, y=0):
        self.char = char
        self.parent = parent
        self.x = x
        self.y = y

        self.left_child = None
        self.right_child = None

def run_demo():
    CHAR_LIST = 'AEOSR INDMU TCLPV GHQBF ZJXKW Y'
    CHAR_POS = {}
    N = 1
    n = 0
    l = 0.2
    letter = Letter(CHAR_LIST[0], x=0.5, y=0.1)
    CHARS = {CHAR_LIST[0]: letter}
    parent_line = [letter]
    current_line = []
    for c in CHAR_LIST[1:]:
        if c==' ':
            continue
        parent = parent_line[n]
        if parent.left_child is None:
            x = parent.x - 0.2/N
            letter = Letter(char=c, parent=parent, x=x, y=l)
            parent.left_child = letter
        else:
            x = parent.x + 0.2/N
            letter = Letter(char=c, parent=parent, x=x, y=l)
            parent.right_child = letter
            n += 1
        CHARS[c] = letter
        current_line += [letter]
        if n == N:
            n = 0
            N *= 2
            l += 0.1
            parent_line = current_line
            current_line = []
    CURRENT_PHRASE_LINE = l+0.1
    threshold_top = 1.1
    threshold_bot = 0.6
    threshold_sides = 0.7
    diagonal_threshold = 0.5*pi/4
    action_threshold = 0.4
    recalibration_threshold = 3
    offsetx = 0
    offsety = 0

    current_key = CHARS[CHAR_LIST[0]]
    last_dir = DIRECTION.NONE
    current_phrase = "_"
    t0 = time.time()
    blink_t0 = t0
    cooldown = False

    screen_width, screen_height = get_screen_size()
        
    args = parse_common_args()

    filter_method = args.filter
    camera_index = args.camera
    calibration_method = args.calibration
    background_path = args.background
    confidence_level = args.confidence
    ema_alpha = args.ema_alpha

    gaze_estimator = GazeEstimator(model_name=args.model)

    if args.model_file and os.path.isfile(args.model_file):
        gaze_estimator.load_model(args.model_file)
        print(f"[demo] Loaded gaze model from {args.model_file}")
    else:
        if calibration_method == "9p":
            run_9_point_calibration(gaze_estimator, camera_index=camera_index)
        elif calibration_method == "5p":
            run_5_point_calibration(gaze_estimator, camera_index=camera_index)
        elif calibration_method == "dense":
            run_dense_grid_calibration(
                gaze_estimator,
                rows=args.grid_rows,
                cols=args.grid_cols,
                margin_ratio=args.grid_margin,
                camera_index=camera_index,
            )
        else:
            run_lissajous_calibration(gaze_estimator, camera_index=camera_index)

    if filter_method == "kalman":
        kalman = make_kalman()
        smoother = KalmanSmoother(kalman)
        smoother.tune(gaze_estimator, camera_index=camera_index)
    elif filter_method == "kalman_ema":
        kalman = make_kalman()
        smoother = KalmanEMASmoother(kalman, ema_alpha=ema_alpha)
        smoother.tune(gaze_estimator, camera_index=camera_index)
    elif filter_method == "kde":
        kalman = None
        smoother = KDESmoother(screen_width, screen_height, confidence=confidence_level)
    else:
        kalman = None
        smoother = NoSmoother()

    if background_path and os.path.isfile(background_path):
        background = cv2.imread(background_path)
        background = cv2.resize(background, (screen_width, screen_height))
    else:
        background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        background[:] = (50, 50, 50)
        for k in CHARS.keys():
            cv2.putText(background,
                        k,
                        (int(CHARS[k].x*screen_width), int(CHARS[k].y*screen_height)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255,255,255),
                        2,
                        cv2.LINE_AA,)

    cam_width, cam_height = 320, 240
    BORDER = 2
    MARGIN = 20
    cursor_alpha = 0.0
    cursor_step = 0.05

    signal_exit = False
    signal_recalibrate = False
    while not signal_exit:
        if signal_recalibrate:
            signal_recalibrate = False
            if calibration_method == "9p":
                run_9_point_calibration(gaze_estimator, camera_index=camera_index)
            elif calibration_method == "5p":
                run_5_point_calibration(gaze_estimator, camera_index=camera_index)
            elif calibration_method == "dense":
                run_dense_grid_calibration(
                    gaze_estimator,
                    rows=args.grid_rows,
                    cols=args.grid_cols,
                    margin_ratio=args.grid_margin,
                    camera_index=camera_index,
                )
            else:
                run_lissajous_calibration(gaze_estimator, camera_index=camera_index)
        with camera(camera_index) as cap, fullscreen("Gaze Estimation"):
            prev_time = time.time()

            for frame in iter_frames(cap):
                features, blink_detected = gaze_estimator.extract_features(frame)

                if blink_detected:
                    x_pred = y_pred = None
                    contours = []
                    cursor_alpha = max(cursor_alpha - cursor_step, 0.0)
                    dir = DIRECTION.BLINK
                    dir_txt_pos = (50, 150)
                elif features is not None:
                    gaze_point = gaze_estimator.predict(np.array([features]))[0]
                    x, y = map(int, gaze_point)
                    x_pred, y_pred = smoother.step(x, y)
                    contours = smoother.debug.get("contours", [])
                    cursor_alpha = min(cursor_alpha + cursor_step, 1.0)
                    dir = DIRECTION.NONE
                else:
                    x_pred = y_pred = None
                    contours = []
                    cursor_alpha = max(cursor_alpha - cursor_step, 0.0)
                    dir = DIRECTION.NO_FACE
                    dir_txt_pos = (50, 150)

                canvas = background.copy()

                if filter_method == "kde" and contours:
                    cv2.drawContours(canvas, contours, -1, (15, 182, 242), 5)

                if x_pred is not None and y_pred is not None and cursor_alpha > 0:
                    x_pred -= offsetx
                    y_pred -= offsety
                    draw_cursor(canvas, x_pred, y_pred, cursor_alpha)

                thumb = make_thumbnail(frame, size=(cam_width, cam_height), border=BORDER)
                h, w = thumb.shape[:2]
                canvas[-h - MARGIN : -MARGIN, -w - MARGIN : -MARGIN] = thumb

                now = time.time()
                fps = 1 / (now - prev_time)
                prev_time = now
                
                if dir==DIRECTION.NONE and x_pred is not None and y_pred is not None:
                    nx = x_pred-screen_width/2
                    anx = abs(nx)
                    ny = y_pred-screen_height/2
                    any = abs(ny)
                    if anx > (threshold_sides*screen_width/2) or ny > (threshold_bot*screen_height/2) or ny < (-threshold_top*screen_height/2):
                        if atan2(min(anx,any), max(anx,any)) > diagonal_threshold:
                            angle = atan2(-ny, nx)
                            if angle > 0:
                                if angle < pi/2:
                                    dir = DIRECTION.NORTHEAST
                                else:
                                    dir = DIRECTION.NORTHWEST
                            else:
                                if -angle < pi/2:
                                    dir = DIRECTION.SOUTHEAST
                                else:
                                    dir = DIRECTION.SOUTHWEST
                        elif abs(nx) > abs(ny):
                            if nx > 0:
                                dir = DIRECTION.RIGHT
                            else:
                                dir = DIRECTION.LEFT
                        else:
                            if ny > 0:
                                dir = DIRECTION.DOWN
                            else:
                                dir = DIRECTION.UP
                    else:
                        dir = DIRECTION.CENTER
                    dir_txt_pos = (50+x_pred, y_pred)

                cv2.putText(
                    canvas,
                    f"FPS: {int(fps)}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                blink_txt = "Blinking" if dir==DIRECTION.BLINK else "No face detected" if dir==DIRECTION.NO_FACE else "Gazing"
                blink_clr = (0, 0, 255) if blink_detected else (0, 255, 0)
                cv2.putText(
                    canvas,
                    blink_txt,
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    blink_clr,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    f"Looking {dir}",
                    dir_txt_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    blink_clr,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    f"Escrevendo: {current_phrase}",
                    (50, int(CURRENT_PHRASE_LINE*screen_height)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    blink_clr,
                    2,
                    cv2.LINE_AA,
                )
                draw_cursor(canvas, current_key.x*screen_width+5, current_key.y*screen_height+5, cursor_alpha)
                canvas = cv2.copyMakeBorder(canvas, 50,50,50,50, cv2.BORDER_CONSTANT, value=COLORS[dir])
                
                tf = now
                dt = tf-t0
                if (last_dir==DIRECTION.NO_FACE or last_dir==DIRECTION.BLINK) and dt > recalibration_threshold:
                    signal_recalibrate = True
                    print(f"Requested recalibration with dir {dir} and dt {dt}")
                    break
                
                elif dir!=last_dir:
                    print(f"Dir changed {last_dir}->{dir} with dt {dt}")
                    t0 = tf
                    
                    if dt > action_threshold:
                        if last_dir==DIRECTION.CENTER:
                            cooldown = False
                        elif last_dir==DIRECTION.BLINK:
                            print(f"Selected key: {current_key.char}")
                            current_phrase = current_phrase[:-1] + current_key.char + '_'
                            current_key = CHARS[CHAR_LIST[0]]
                            cooldown = True
                        elif last_dir==DIRECTION.UP:
                            if current_key.parent is not None:
                                current_key = current_key.parent
                                cooldown = True
                        elif last_dir==DIRECTION.LEFT:
                            if current_key.left_child is not None:
                                current_key = current_key.left_child
                                cooldown = True
                        elif last_dir==DIRECTION.RIGHT:
                            if current_key.right_child is not None:
                                current_key = current_key.right_child
                                cooldown = True
                        elif last_dir==DIRECTION.DOWN:
                            current_key = CHARS[CHAR_LIST[0]]
                            current_phrase = current_phrase[:-2] + '_'
                            cooldown = True
                        elif last_dir==DIRECTION.SOUTHEAST:
                            tts = gTTS(text=current_phrase, lang='pt-br')
                            mp3_as_bytes = next(tts.stream())
                            audio = AudioSegment.from_file(BytesIO(mp3_as_bytes), format="mp3")
                            play(audio)
                last_dir = dir

                cv2.imshow("Gaze Estimation", canvas)
                keyboard_pressed = cv2.waitKey(1)
                if keyboard_pressed == 27:
                    signal_exit = True
                    break
                elif keyboard_pressed==81:
                    offsetx -= 1
                elif keyboard_pressed==82:
                    offsety -= 1
                elif keyboard_pressed==83:
                    offsetx += 1
                elif keyboard_pressed==84:
                    offsety += 1
                else:
                    print(f"Keyboard pressed: {keyboard_pressed}")


if __name__ == "__main__":
    run_demo()