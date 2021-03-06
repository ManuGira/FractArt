import os
import cv2 as cv
import numpy as np

def color_hex2rgb(hexa):
    # hexa must be in format #123456 in hexadecimal
    r, g, b = hexa[1:3], hexa[3:5], hexa[5:7]
    r, g, b = [int(c, 16) for c in [r, g, b]]
    return r, g, b


def sec_to_hms(sec):
    h = int(sec / 60 / 60)
    m = int(sec / 60 - h * 60)
    s = int(sec - h * 60 * 60 - m * 60)
    return h, m, s


def pth(*args):
    out = "."
    for arg in args:
        if isinstance(arg, list):
            arg = os.path.join(*arg)
        out = os.path.join(out, arg)
    out = os.path.normpath(out)
    out = os.path.join(".", out)
    return out

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

