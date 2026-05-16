import sys

import os
import time
from math import atan2, pi

from enum import StrEnum, auto

import cv2
import numpy as np
import screeninfo

from Dictionary import Dictionary, Entry, generate_dictionary
from utils import draw_background

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
# from eyetrax.gaze import GazeEstimator
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

def run_demo():
    cv2.namedWindow("thumbnail")
    cv2.setWindowProperty("thumbnail", cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_AUTOSIZE)
    
    alphabet = generate_dictionary("alphabet.json")
    
    CURRENT_PHRASE_LINE = alphabet.y_start+alphabet.heigth+0.1
    threshold_top = 0.9
    threshold_bot = 0.5
    threshold_sides = 0.7
    diagonal_threshold = 0.5*pi/4
    action_threshold = 0.4
    recalibration_threshold = 3
    offsetx = 0
    offsety = 0
    Sx = -10
    Sy = 10

    current_key = alphabet.lines[0][0]
    last_dir = DIRECTION.NONE
    current_phrase = "_"
    t0 = time.time()
    cooldown = False
        
    args = parse_common_args()

    filter_method = args.filter
    camera_index = args.camera
    screen_index = args.screen
    camera_rotate = args.camera_rotate
    calibration_method = args.calibration
    background_path = args.background
    confidence_level = args.confidence
    ema_alpha = args.ema_alpha
    kde_draw_contours = args.kde_draw_contours
    show_background = args.show_background

    screen_width, screen_height = get_screen_size(screen_index=screen_index)

    if background_path and os.path.isfile(background_path):
        background = cv2.imread(background_path)
        background = cv2.resize(background, (screen_width, screen_height))
    else:
        background = draw_background(alphabet, screen_width, screen_height, Sx, Sy, font_scale=1.4)
    
    if show_background:
        fullscreen("Gaze Estimation")
        screen = screeninfo.get_monitors()[screen_index]
        cv2.moveWindow("Gaze Estimation", screen.x-1, screen.y-1)
        cv2.setWindowProperty("Gaze Estimation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Gaze Estimation", background)
        while cv2.waitKey(400) != 27:
            pass
        exit(0)

    gaze_estimator = GazeEstimator(model_name=args.model)

    if args.model_file and os.path.isfile(args.model_file):
        gaze_estimator.load_model(args.model_file)
        print(f"[demo] Loaded gaze model from {args.model_file}")
    else:
        if calibration_method == "9p":
            ret = run_9_point_calibration(gaze_estimator, camera_index=camera_index, screen_index=screen_index, camera_rotate=camera_rotate)
        elif calibration_method == "5p":
            ret = run_5_point_calibration(gaze_estimator, camera_index=camera_index, screen_index=screen_index, camera_rotate=camera_rotate)
        elif calibration_method == "dense":
            ret = run_dense_grid_calibration(
                gaze_estimator,
                rows=args.grid_rows,
                cols=args.grid_cols,
                margin_ratio=args.grid_margin,
                camera_index=camera_index,
                screen_index=screen_index,
                camera_rotate=camera_rotate,
            )
        else:
            ret = run_lissajous_calibration(gaze_estimator, camera_index=camera_index, screen_index=screen_index, camera_rotate=camera_rotate)
        if not ret:
            exit()

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

    cam_width, cam_height = 320, 240
    BORDER = 2
    MARGIN = 20
    cursor_alpha = 0.0
    cursor_step = 0.05

    signal_exit = False
    signal_recalibrate = False
    while not signal_exit:
        if signal_recalibrate:
            if calibration_method == "9p":
                ret = run_9_point_calibration(gaze_estimator, camera_index=camera_index, screen_index=screen_index, camera_rotate=camera_rotate)
            elif calibration_method == "5p":
                ret = run_5_point_calibration(gaze_estimator, camera_index=camera_index, screen_index=screen_index, camera_rotate=camera_rotate)
            elif calibration_method == "dense":
                ret = run_dense_grid_calibration(
                    gaze_estimator,
                    rows=args.grid_rows,
                    cols=args.grid_cols,
                    margin_ratio=args.grid_margin,
                    camera_index=camera_index,
                    screen_index=screen_index,
                    camera_rotate=camera_rotate,
                )
            else:
                ret = run_lissajous_calibration(gaze_estimator, camera_index=camera_index, screen_index=screen_index, camera_rotate=camera_rotate)
            if not ret:
                exit()
            signal_recalibrate = False
            t0 = time.time()
        with camera(camera_index) as cap, fullscreen("Gaze Estimation"):
            prev_time = time.time()

            for frame in iter_frames(cap):
                if camera_rotate == 1:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif camera_rotate == 2:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
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

                # if filter_method == "kde" and contours and kde_draw_contours:
                #     cv2.drawContours(canvas, contours, -1, (15, 182, 242), 5)

                if x_pred is not None and y_pred is not None and cursor_alpha > 0:
                    x_pred -= offsetx
                    y_pred -= offsety
                    draw_cursor(canvas, x_pred, y_pred, cursor_alpha)

                cv2.imshow("thumbnail", frame)
                # thumb = make_thumbnail(frame, size=(cam_width, cam_height), border=BORDER)
                # h, w = thumb.shape[:2]
                # canvas[-h - MARGIN : -MARGIN, -w - MARGIN : -MARGIN] = thumb

                now = time.time()
                fps = 1 / (now - prev_time)
                prev_time = now
                
                if dir==DIRECTION.NONE and x_pred is not None and y_pred is not None:
                    nx = x_pred-screen_width/2
                    anx = abs(nx)
                    ny = y_pred-screen_height/2
                    any = abs(ny)
                    if anx > (threshold_sides*screen_width/2) or ny > (threshold_bot*screen_height/2) or ny < (-threshold_top*screen_height/2):
                        # if atan2(min(anx,any), max(anx,any)) > diagonal_threshold:
                        if anx > screen_width/2 and any > screen_height/2:
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
                # cv2.putText(
                #     canvas,
                #     f"Looking {dir}",
                #     dir_txt_pos,
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     1.2,
                #     blink_clr,
                #     2,
                #     cv2.LINE_AA,
                # )
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
                cv2.putText(
                    canvas,
                    f"VERMELHO = CIMA/BAIXO   VERDE = LADOS   AZUL = DIAGONAIS",
                    (50, int(CURRENT_PHRASE_LINE*screen_height)+100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    blink_clr,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    f"CIMA = SUBIR   BAIXO = APAGAR",
                    (50, int(CURRENT_PHRASE_LINE*screen_height)+150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    blink_clr,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    f"PISCAR 1 SEGUNDO = ESCREVER LETRA",
                    (50, int(CURRENT_PHRASE_LINE*screen_height)+200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    blink_clr,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    f"PISCAR 3 SEGUNDOS = RECALIBRAR",
                    (50, int(CURRENT_PHRASE_LINE*screen_height)+250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    blink_clr,
                    2,
                    cv2.LINE_AA,
                )
                draw_cursor(canvas, current_key.x*screen_width-Sx, current_key.y*screen_height-Sy, cursor_alpha)
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
                        # elif last_dir==DIRECTION.SOUTHEAST:
                        #     tts = gTTS(text=current_phrase, lang='pt-br')
                        #     mp3_as_bytes = next(tts.stream())
                        #     audio = AudioSegment.from_file(BytesIO(mp3_as_bytes), format="mp3")
                        #     play(audio)
                        # elif last_dir==DIRECTION.SOUTHWEST:
                        #     current_phrase = '_'
                last_dir = dir

                screen = screeninfo.get_monitors()[screen_index]
                cv2.moveWindow("Gaze Estimation", screen.x-1, screen.y-1)
                cv2.setWindowProperty("Gaze Estimation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow("Gaze Estimation", canvas)
                keyboard_pressed = cv2.waitKey(1)
                print(f"Keyboard pressed: {keyboard_pressed}")
                if keyboard_pressed == 27:
                    signal_exit = True
                    break
                elif keyboard_pressed==81:
                    offsetx -= 10
                    print(f"offx {offsetx}")
                elif keyboard_pressed==82:
                    offsety -= 10
                    print(f"offy {offsety}")
                elif keyboard_pressed==83:
                    offsetx += 10
                    print(f"offx {offsetx}")
                elif keyboard_pressed==84:
                    offsety += 10
                    print(f"offy {offsety}")
                else:
                    pass

import traceback
if __name__ == "__main__":
    try:
        run_demo()
    except:
        print(traceback.format_exc())