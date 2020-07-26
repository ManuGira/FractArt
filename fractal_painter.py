import numpy as np
from numba import njit
import cv2 as cv


@njit
def apply_color_map(hits, color_map):
    H, W = hits.shape
    out = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    for j in range(H):
        for i in range(W):
            bgr = color_map[hits[j, i]]
            out[j, i, :] = bgr
    return out


def color_gradient(colors_bgr, length, interpolation=cv.INTER_LINEAR):
    colors_bgr = np.array(colors_bgr, dtype=np.uint8)
    return cv.resize(colors_bgr, dsize=(3, length), interpolation=interpolation)


def color_map(hits):
    # dB = lambda x: (20 * np.log(x + 1)).astype(np.uint16)
    # hits = dB(hits)
    N = 1024
    # BRG colors
    color_map_0 = color_gradient([
        [0, 0, 0],
        [255, 255, 95],
        [0, 0, 0],
        [169, 127, 255],
        [0, 0, 0],
        [127, 255, 255],
        [0, 0, 0],
    ], N)
    color_map_1 = color_gradient([
        [0, 0, 255],
        [255, 255, 255],
    ], N)
    color_switch = np.zeros(shape=(N, 1))
    # color_switch[int(light_effect * N)] = 1
    color_map = color_map_0 * (1 - color_switch) + color_map_1 * color_switch
    # if self.light_effect >= 0:
    #     color_map[int(maxhit*self.light_effect), :] = [255, 255, 255]
    color_map.shape = (N, 1, 3)
    return apply_color_map(hits, color_map)


def glow_effect(img):
    img2 = cv.GaussianBlur(img, ksize=(21, 21), sigmaX=5, sigmaY=5)
    out = img2
    out[img>img2] = img[img>img2]
    return out