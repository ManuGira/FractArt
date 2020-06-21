import os
import cv2 as cv
import numpy as np

def export_to_png(name, data):
    folder = "gallery"
    if not os.path.exists(folder):
        os.mkdir(folder)
    file = os.path.join(folder, f"{name}.png")
    cv.imwrite(file, data)

def convert_to_show(img, reshape=None):
    img = np.array(img).copy()
    if reshape is not None:
        img = np.reshape(img, newshape=reshape)

    # map False,True to 0,255
    if img.dtype == np.bool:
        img8 = np.zeros_like(img, dtype=np.uint8)
        img8[img] = 255
        img = img8

    # force color image
    if len(img.shape) == 3 and not img.shape[2]==3:
        img = img[:, :, 0]
        img.shape = img.shape[0:2]
    if not len(img.shape) == 3:
        img.shape += (1,)
        img = np.repeat(img, repeats=3, axis=2)

    return img

def draw_poly(img, pts, fill=False, color=127, reshape=None):
    img = convert_to_show(img, reshape=reshape)

    pts = np.squeeze(pts)
    if len(pts.shape) == 2:
        pts = [pts]
    pts = np.array(np.round(pts), dtype=np.int)

    color = np.squeeze(color)
    if len(color.shape) == 0:
        color = (int(color),)*3

    if fill:
        img = cv.fillPoly(img, pts, color)
    else:
        img = cv.polylines(img, pts, True, color=color)
    return img

def imshow(img, name="utils", ms=0, reshape=None):
    toshow = convert_to_show(img, reshape=reshape)
    cv.imshow(name, toshow)
    cv.waitKey(ms)

