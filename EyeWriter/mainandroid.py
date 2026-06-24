import sys

import os
import time
from math import atan2, pi
import traceback

from enum import StrEnum, auto

import cv2
import numpy as np
import screeninfo

from kivy.app import App
# from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
# from kivy.uix.widget import Widget
# from kivy.graphics import Color
from kivy.graphics import Rectangle
from kivy.clock import Clock
from kivy.uix.camera import Camera
from kivy.core.image import Texture
import kivy

from kivy.utils import platform
print(platform)
if platform != 'linux':
    from android.permissions import request_permissions, Permission

from Dictionary import (
    Dictionary,
    Entry,
    load_dictionary_from_json
)
from utils import draw_background

from eyetrax.calibration import (
    prepare_5_point_calibration, run_5_point_calibration,
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
from eyetrax.utils.draw import (
    draw_cursor,
    make_thumbnail
)
from eyetrax.utils.screen import get_screen_size
from eyetrax.utils.video import camera, fullscreen, iter_frames

# Audio playing from https://stackoverflow.com/questions/76696178/how-to-make-play-a-sound-from-a-string
# from pydub.playback import play
# from pydub import AudioSegment
# from gtts import gTTS
# from io import BytesIO


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
    DIRECTION.CENTER: (0, 0, 0),
    DIRECTION.UP: (0, 0, 150),
    DIRECTION.DOWN: (0, 0, 150),
    DIRECTION.LEFT: (0, 150, 0),
    DIRECTION.RIGHT: (0, 150, 0),
    DIRECTION.NORTHEAST: (150, 0, 0),
    DIRECTION.NORTHWEST: (150, 0, 0),
    DIRECTION.SOUTHEAST: (150, 0, 0),
    DIRECTION.SOUTHWEST: (150, 0, 0),
    DIRECTION.BLINK: (255, 255, 255),
    DIRECTION.NO_FACE: (150, 0, 150),
    DIRECTION.NONE: (150, 150, 150)
}


class DrawingCanvas(BoxLayout):
    class APPSTATE(StrEnum):
        PREPARING_CALIBRATION = auto()
        CALIBRATING = auto()
        RUNNING = auto()

    state = APPSTATE.PREPARING_CALIBRATION
    cam = None
    alphabet:Dictionary = None

    threshold_top = 0.9
    threshold_bot = 0.5
    threshold_sides = 0.7
    # diagonal_threshold = 0.5*pi/4
    action_threshold = 0.4
    recalibration_threshold = 3
    offsetx = 0
    offsety = 0
    # Sx = -10
    # Sy = 10

    cam_width, cam_height = 640, 480
    BORDER = 2
    MARGIN = 20
    cursor_alpha = 0.0
    cursor_step = 0.05
    cooldown = False
    last_dir = DIRECTION.NONE
    current_phrase = "_"

    signal_exit = False
    signal_recalibrate = False

    def __init__(self, **kwargs):
        super(DrawingCanvas, self).__init__(**kwargs)

        self.alphabet = load_dictionary_from_json("src/main/assets/alphabet.json")
        self.background = draw_background(self.alphabet,
                                          1080, int(1080*self.height/self.width),
                                          font_scale=1)
        self.update_canvas(self.background)
        self.cam = Camera(play=True, resolution=(self.cam_width, self.cam_height))

        self.CURRENT_PHRASE_LINE = self.alphabet.y_start+self.alphabet.height+0.1
        self.current_key:Entry = self.alphabet.lines[0][0]
        self.t0 = time.time()

        self.args = parse_common_args()

        self.filter_method = self.args.filter
        self.camera_index = self.args.camera
        self.screen_index = self.args.screen
        self.camera_rotate = self.args.camera_rotate
        self.calibration_method = self.args.calibration
        self.confidence_level = self.args.confidence
        self.ema_alpha = self.args.ema_alpha

        self.gaze_estimator = GazeEstimator(model_name=self.args.model)

        if self.args.model_file and os.path.isfile(self.args.model_file):
            self.gaze_estimator.load_model(self.args.model_file)
            print(f"[demo] Loaded gaze model from {self.args.model_file}")

        if self.filter_method == "kalman":
            kalman = make_kalman()
            smoother = KalmanSmoother(kalman)
            smoother.tune(self.gaze_estimator, camera_index=self.camera_index)
        elif self.filter_method == "kalman_ema":
            kalman = make_kalman()
            smoother = KalmanEMASmoother(kalman, ema_alpha=self.ema_alpha)
            smoother.tune(self.gaze_estimator, camera_index=self.camera_index)
        elif self.filter_method == "kde":
            kalman = None
            smoother = KDESmoother(self.width,
                                   self.height,
                                   confidence=self.confidence_level)
        else:
            kalman = None
            smoother = NoSmoother()

        self.prev_time = time.time()

    def prepare_calibration(self):
        if self.calibration_method == "5p":
            canvas = prepare_5_point_calibration(
                self.width,
                self.height)
        return canvas

    def run_calibration(self, frame):
        if self.calibration_method == "9p":
            ret = run_9_point_calibration(
                self.gaze_estimator,
                camera_index=self.camera_index,
                screen_index=self.screen_index,
                camera_rotate=self.camera_rotate)
        elif self.calibration_method == "5p":
            ok, canvas = run_5_point_calibration(
                self.gaze_estimator, frame)
        elif self.calibration_method == "dense":
            ret = run_dense_grid_calibration(
                self.gaze_estimator,
                rows=self.args.grid_rows,
                cols=self.args.grid_cols,
                margin_ratio=self.args.grid_margin,
                camera_index=self.camera_index,
                screen_index=self.screen_index,
                camera_rotate=self.camera_rotate)
        else:
            ret = run_lissajous_calibration(
                self.gaze_estimator,
                camera_index=self.camera_index,
                screen_index=self.screen_index,
                camera_rotate=self.camera_rotate)
        return ok, canvas

    def get_frame(self):
        frame = self.cam.texture
        frame = np.frombuffer(frame.pixels, np.uint8).reshape(
            frame.height,
            frame.width,
            -1)
        frame = frame[:, :, 0:3]

        # if camera_rotate == 1:
        #     frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # elif camera_rotate == 2:
        #     frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return frame

    def get_direction(self, dir, x_pred, y_pred):
        if dir == DIRECTION.NONE and x_pred is not None and y_pred is not None:
            nx = x_pred - self.width/2
            anx = abs(nx)
            ny = y_pred - self.height/2
            any = abs(ny)
            if anx > (self.threshold_sides * self.width/2) or ny > (self.threshold_bot * self.height/2) or ny < (-self.threshold_top*self.height/2):
                # if atan2(min(anx,any), max(anx,any)) > diagonal_threshold:
                if anx > self.width/2 and any > self.height/2:
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
        return dir

    def paint_canvas(self, x_pred, y_pred, dir, blink_detected, cursor_alpha, fps):
        canvas = self.background.copy()
        if x_pred is not None and y_pred is not None and cursor_alpha > 0:
            x_pred -= self.offsetx
            y_pred -= self.offsety
            draw_cursor(canvas, x_pred, y_pred, cursor_alpha)
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
            f"Escrevendo: {self.current_phrase}",
            (50, int(self.CURRENT_PHRASE_LINE*self.height)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            blink_clr,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "VERMELHO = CIMA/BAIXO   VERDE = LADOS   AZUL = DIAGONAIS",
            (50, int(self.CURRENT_PHRASE_LINE*self.height)+100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            blink_clr,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "CIMA = SUBIR   BAIXO = APAGAR",
            (50, int(self.CURRENT_PHRASE_LINE*self.height)+150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            blink_clr,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "PISCAR 1 SEGUNDO = ESCREVER LETRA",
            (50, int(self.CURRENT_PHRASE_LINE*self.height)+200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            blink_clr,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "PISCAR 3 SEGUNDOS = RECALIBRAR",
            (50, int(self.CURRENT_PHRASE_LINE*self.height)+250),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            blink_clr,
            2,
            cv2.LINE_AA,
        )
        draw_cursor(canvas,
                    self.current_key.x*self.width,
                    self.current_key.y*self.height,
                    cursor_alpha)
        canvas = cv2.copyMakeBorder(canvas,
                                    50, 50, 50, 50,
                                    cv2.BORDER_CONSTANT,
                                    value=COLORS[dir])
        self.update_canvas(canvas)

    def update_canvas(self, new_canvas, frame=None):
        with self.canvas:
            if frame is not None:
                thumbnail = make_thumbnail(frame)
                h, w = thumbnail.shape[:2]
                new_canvas[-h - self.MARGIN : -self.MARGIN, -w - self.MARGIN : -self.MARGIN] = thumbnail
            new_canvas = cv2.flip(new_canvas, 1)
            new_canvas = cv2.rotate(new_canvas, cv2.ROTATE_180)
            self.canvas.clear()
            texture = Texture.create((new_canvas.shape[1], new_canvas.shape[0]),
                                     colorfmt='rgb')
            texture.blit_buffer(new_canvas.flatten(),
                                colorfmt='rgb',
                                bufferfmt='ubyte')
            Rectangle(texture=texture,
                      pos=self.pos,
                      size=(self.width, self.height))

    def run_smoother(self):
        # if self.filter_method == "kalman":
        #     self.kalman = make_kalman()
        #     self.smoother = KalmanSmoother(kalman)
        #     self.smoother.tune(gaze_estimator, camera_index=camera_index)
        # elif self.filter_method == "kalman_ema":
        #     self.kalman = make_kalman()
        #     self.smoother = KalmanEMASmoother(kalman, ema_alpha=ema_alpha)
        #     self.smoother.tune(gaze_estimator, camera_index=camera_index)
        # el
        if self.filter_method == "kde":
            self.kalman = None
            self.smoother = KDESmoother(self.width,
                                        self.height,
                                        confidence=self.confidence_level)
        # else:
        #     self.kalman = None
        #     self.smoother = NoSmoother()

    def update(self, event):
        if self.state == self.APPSTATE.PREPARING_CALIBRATION:
            canvas = self.prepare_calibration()
            self.update_canvas(canvas)
            self.state = self.APPSTATE.CALIBRATING
            Clock.schedule_once(self.update, 0.033)
            return

        frame = self.get_frame()
        if self.state == self.APPSTATE.CALIBRATING:
            ok, canvas = self.run_calibration(frame)
            self.update_canvas(canvas)
            if ok:
                self.state = self.APPSTATE.RUNNING
                self.t0 = time.time()
                self.run_smoother()
            Clock.schedule_once(self.update, 0.033)
            return

        features, blink_detected = self.gaze_estimator.extract_features(frame)

        if blink_detected:
            x_pred = y_pred = None
            contours = []
            cursor_alpha = max(self.cursor_alpha - self.cursor_step, 0.0)
            dir = DIRECTION.BLINK
            dir_txt_pos = (50, 150)
        elif features is not None:
            gaze_point = self.gaze_estimator.predict(np.array([features]))[0]
            x, y = map(int, gaze_point)
            x_pred, y_pred = self.smoother.step(x, y)
            contours = self.smoother.debug.get("contours", [])
            cursor_alpha = min(self.cursor_alpha + self.cursor_step, 1.0)
            dir = DIRECTION.NONE
            dir_txt_pos = (50 + x_pred, y_pred)
        else:
            x_pred = y_pred = None
            contours = []
            cursor_alpha = max(self.cursor_alpha - self.cursor_step, 0.0)
            dir = DIRECTION.NO_FACE
            dir_txt_pos = (50, 150)

        # if filter_method == "kde" and contours and kde_draw_contours:
        #     cv2.drawContours(canvas, contours, -1, (15, 182, 242), 5)

        now = time.time()
        fps = 1 / (now - self.prev_time)
        self.prev_time = now

        dir = self.get_direction(dir, x_pred, y_pred)
        self.paint_canvas(x_pred, y_pred, dir, blink_detected, cursor_alpha, fps)

        tf = now
        dt = tf-self.t0
        if (self.last_dir == DIRECTION.NO_FACE or self.last_dir == DIRECTION.BLINK) and dt > self.recalibration_threshold:
            self.state = self.APPSTATE.PREPARING_CALIBRATION
            print(f"Requested recalibration with dir {dir} and dt {dt}")
            Clock.schedule_once(self.update, 0.033)
            return

        elif dir != self.last_dir:
            print(f"Dir changed {self.last_dir}->{dir} with dt {dt}")
            self.t0 = tf

            if dt > self.action_threshold:
                if self.last_dir == DIRECTION.CENTER:
                    cooldown = False
                elif self.last_dir == DIRECTION.BLINK:
                    print(f"Selected key: {self.current_key.key}")
                    current_phrase = self.current_phrase[:-1] + self.current_key.key + '_'
                    current_key = self.alphabet.lines[0][0]
                    cooldown = True
                elif self.last_dir == DIRECTION.UP:
                    if current_key.parent is not None:
                        current_key = current_key.parent
                        cooldown = True
                elif self.last_dir == DIRECTION.LEFT:
                    if current_key.left_child is not None:
                        current_key = current_key.left_child
                        cooldown = True
                elif self.last_dir == DIRECTION.RIGHT:
                    if current_key.right_child is not None:
                        current_key = current_key.right_child
                        cooldown = True
                elif self.last_dir == DIRECTION.DOWN:
                    current_key = self.alphabet.lines[0][0]
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
        Clock.schedule_once(self.update, 0.033)

        # keyboard_pressed = cv2.waitKey(1)
        # print(f"Keyboard pressed: {keyboard_pressed}")
        # if keyboard_pressed == 27:
        #     signal_exit = True
        #     return
        # elif keyboard_pressed == 81:
        #     self.offsetx -= 10
        #     print(f"offx {self.offsetx}")
        # elif keyboard_pressed == 82:
        #     self.offsety -= 10
        #     print(f"offy {self.offsety}")
        # elif keyboard_pressed == 83:
        #     self.offsetx += 10
        #     print(f"offx {self.offsetx}")
        # elif keyboard_pressed == 84:
        #     self.offsety += 10
        #     print(f"offy {self.offsety}")
        # else:
        #     pass


class MainApp(App):
    def build(self):
        if platform == 'android':
            request_permissions([
                Permission.CAMERA,
                Permission.WRITE_EXTERNAL_STORAGE,
                Permission.READ_EXTERNAL_STORAGE
            ])

        drawing_canvas = DrawingCanvas()
        # Clock.schedule_interval(drawing_canvas.update, 0.033)
        Clock.schedule_once(drawing_canvas.update, 0.3)
        return drawing_canvas


def run_android():
    MainApp().run()


if __name__ == "__main__":
    try:
        run_android()
    except Exception:
        print(traceback.format_exc())
