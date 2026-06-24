import numpy as np
import cv2

from Dictionary import Dictionary, Entry

SCALE_2_SCREEN_FACTOR = 20

def draw_background(dictionary: Dictionary, screen_width, screen_height, font_scale=1.4, background_color=(50,50,50)):
    Sx = Sy = SCALE_2_SCREEN_FACTOR/2

    background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    background[:] = background_color

    for c in dictionary.entries.values():
        c_pos = (int(c.x*screen_width), int(c.y*screen_height))
        lc = c.left_child
        if lc is not None:
            cv2.line(background,
                     c_pos,
                     (int(lc.x*screen_width), int(lc.y*screen_height)),
                     (255,255,255),
                     1,
                     cv2.LINE_AA)
        rc = c.right_child
        if rc is not None:
            cv2.line(background,
                     c_pos,
                     (int(rc.x*screen_width), int(rc.y*screen_height)),
                     (255,255,255),
                     1,
                     cv2.LINE_AA)

    for c in dictionary.entries.values():
        c_pos = (int(c.x*screen_width), int(c.y*screen_height))
        cv2.ellipse(background,
                    (c_pos, (int(font_scale*SCALE_2_SCREEN_FACTOR*len(c.key)),int(font_scale*SCALE_2_SCREEN_FACTOR)), 0),
                    background_color,
                    int(font_scale*SCALE_2_SCREEN_FACTOR))

    for c in dictionary.entries.values():
        c_pos = (int(c.x*screen_width - SCALE_2_SCREEN_FACTOR*font_scale/2), int(c.y*screen_height + SCALE_2_SCREEN_FACTOR*font_scale/2))
        cv2.putText(img=background,
                    text=c.key,
                    org=c_pos,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale,
                    color=(255,255,255),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                    )

    return background
