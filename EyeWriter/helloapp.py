# import os
# import sys

# import time
# from enum import StrEnum, auto
# from math import atan2, pi
import traceback

import cv2
import numpy as np
# import screeninfo

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
# import time

# from eyetrax.utils.video import fullscreen
# from kivy.uix import camera
# from eyetrax.utils.screen import get_screen_size

# from Dictionary import Dictionary, Entry, load_dictionary_from_json
# from utils import draw_background


class DrawingCanvas(BoxLayout):
    state = 0
    cam = None
    alphabet = None
    cam_mode = None

    def __init__(self, **kwargs):
        super(DrawingCanvas, self).__init__(**kwargs)
        # background = draw_background(self.alphabet,
        #                              self.width, self.height,
        #                              font_scale=1.4)
        # self.alphabet = load_dictionary_from_json("alphabet.json")
        self.background = np.zeros((self.height, self.width, 3),
                                   dtype=np.uint8)
        self.background[:] = (50, 50, 50)

        try:
            self.cam = Camera(play=True, resolution=(640, 480), index=1)
        except:
            self.cam = Camera(play=True, resolution=(640, 480), index=0)

    def update(self, event):
        with self.canvas:
            img = self.cam
            frame = img.texture
            img = np.frombuffer(frame.pixels, np.uint8).reshape(frame.height,
                                                                frame.width,
                                                                -1)
            img = img[:, :, 0:3]

            img = cv2.flip(img, 1)

            camera_rotate = 2
            if camera_rotate == 1:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif camera_rotate == 2:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            self.canvas.clear()
            texture = Texture.create((img.shape[1], img.shape[0]),
                                     colorfmt='rgb')
            texture.blit_buffer(img.flatten(),
                                colorfmt='rgb',
                                bufferfmt='ubyte')

            Rectangle(texture=texture,
                      pos=self.pos,
                      size=(self.width, self.height))
            # Color(1., 0, 0)
            # Rectangle(pos=(10, 50), size=(50, 50))


class HelloApp(App):
    def build(self):
        if platform != 'linux':
            request_permissions([
                Permission.CAMERA,
                Permission.WRITE_EXTERNAL_STORAGE,
                Permission.READ_EXTERNAL_STORAGE
            ])

        drawing_canvas = DrawingCanvas()
        Clock.schedule_interval(drawing_canvas.update, 0.1)
        return drawing_canvas


if __name__ == "__main__":
    try:
        HelloApp().run()
    except Exception:
        print(traceback.format_exc())
