import numpy as np
import cv2

from Dictionary import Dictionary, Entry

SS = 20

def draw_background(dictionary: Dictionary, screen_width, screen_height, Sx, Sy, font_scale=1.4):
    background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    background[:] = (50, 50, 50)
    for c in dictionary.entries.values():
        c_pos = (int(c.x*screen_width), int(c.y*screen_height))
        c_posS = (int(c.x*screen_width)-Sx, int(c.y*screen_height)-Sy)
        lc = c.left_child
        if lc is not None:
            cv2.line(background,
                        c_posS,
                        (int(lc.x*screen_width)-Sx, int(lc.y*screen_height)-Sy),
                        (255,255,255),
                        1,
                        cv2.LINE_AA)
        rc = c.right_child
        if rc is not None:
            cv2.line(background,
                        c_posS,
                        (int(rc.x*screen_width)-Sx, int(rc.y*screen_height)-Sy),
                        (255,255,255),
                        1,
                        cv2.LINE_AA)
            
    for c in dictionary.entries.values():
        c_pos = (int(c.x*screen_width), int(c.y*screen_height))
        c_posS = (int(c.x*screen_width)-Sx, int(c.y*screen_height)-Sy)
        cv2.ellipse(background,
                    (c_posS, (int(font_scale*SS*len(c.key)),int(font_scale*SS)), 0),
                    (50,50,50),
                    int(font_scale*SS))
    
    for c in dictionary.entries.values():
        c_pos = (int(c.x*screen_width), int(c.y*screen_height))
        c_posS = (int(c.x*screen_width)-Sx, int(c.y*screen_height)-Sy)
        cv2.putText(background,
                    c.key,
                    c_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4,
                    (255,255,255),
                    2,
                    cv2.LINE_AA,)
        
    return background