from glob import glob


from pathaia.util.types import Slide

import json
import numpy as np
import base64
import cv2


def export_img_cv2(img):
    _, buff = cv2.imencode(".png", img)
    return base64.b64encode(buff).decode("ascii")


slide_thum = [
    export_img_cv2(np.array(Slide(p).get_thumbnail((50, 50))))
    for p in glob("/media/AprioricsSlides" + "/*" + "-1-??-1_*")
]

import json

with open("test", "w") as fp:
    json.dump(slide_thum, fp)
