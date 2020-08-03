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


def color_map(hits, max_iter):
    # dB = lambda x: (20 * np.log(x + 1)).astype(np.uint16)
    # hits = dB(hits)
    # BRG colors
    N = 8192
    color_list = [
        [0, 0, 0],
        [255, 255, 95],
        [0, 0, 0],
        [169, 127, 255],
        [0, 0, 0],
        [127, 255, 255],
        [0, 0, 0],
    ]*8
    color_map_0 = color_gradient(color_list, N)
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

def cvtBRG_to_HLScube(bgr):
    hls = cv.cvtColor(bgr, cv.COLOR_BGR2HLS)
    hls = hls.astype(np.float)
    hls[:, :, 0] /= 180
    hls[:, :, 1] /= 255
    hls[:, :, 2] /= 255
    return hls

def cvtHLScube_to_BGR(hls):
    hls[:, :, 0] *= 180
    hls[:, :, 1] *= 255
    hls[:, :, 2] *= 255
    hls = hls.astype(np.uint8)
    bgr = cv.cvtColor(hls, cv.COLOR_HLS2BGR)
    return bgr

def neon_effect(hits, level, color_bgr, width=200, brigther_factor=10, glow_size1=3):
    H, W, = hits.shape
    mask_gray = np.zeros(shape=(H, W), dtype=np.uint8)
    mask = np.logical_and(level - width < hits, hits < level + width)
    mask_gray[mask] = 255

    light = mask_gray.astype(np.uint16)*brigther_factor
    glow = mask_gray.copy()
    for k in range(20):
        glow = cv.GaussianBlur(glow, ksize=(0, 0), sigmaX=glow_size1, sigmaY=0)
        light += glow
    light = light // brigther_factor
    light[light > 255] = 255
    light = light.astype(np.uint8)
    light.shape += (1,)

    # bgr to hls, shape (1,1,3)
    hls = cv.cvtColor(np.array([[color_bgr]], dtype=np.uint8), cv.COLOR_BGR2HLS)  # uint8 (1x1x3)
    zeros = np.zeros(shape=(H, W, 1), dtype=np.uint8)
    mask_hls = np.concatenate((
        zeros + hls[0, 0, 0],
        light,
        zeros + hls[0, 0, 2],
    ), axis=2)

    mask_bgra = np.concatenate((
        cv.cvtColor(mask_hls, cv.COLOR_HLS2BGR),
        light,
    ), axis=2)
    return mask_bgra


def blend(bgras):
    bgrs = []
    alphas = []
    for bgra in bgras:
        bgra = bgra.astype(float)
        alpha = np.repeat(bgra[:, :, 3:4]/255, 3, axis=2)
        bgr = cv.multiply(bgra[:, :, :3], alpha)

        bgrs.append(bgr)
        alphas.append(alpha)

    out_bgr = bgrs[0]
    for bgr in bgrs[1:]:
        out_bgr = cv.add(out_bgr, bgr)
    out_bgr[out_bgr > 255] = 255
    out_bgr = out_bgr.astype(np.uint8)

    out_alpha = 1-alphas[0][:, :, 0:1]
    for alpha in alphas[1:]:
        out_alpha = 1-alpha[:, :, 0:1]
    out_alpha = 1-out_alpha
    out_alpha = (out_alpha*255).astype(np.uint8)

    out = np.concatenate((out_bgr, out_alpha), axis=2)
    return out


def main():
    import os
    from utils import pth
    import pickle
    import time

    winname = "image"
    def nothing(x):
        pass

    def createTrackbars(winname, trackbars):
        for name, val in trackbars.items():
            cv.createTrackbar(name, winname, val[0], val[1], nothing)

    def getTrackbars(winname, trackbars):
        out = []
        for name in trackbars.keys():
            val = cv.getTrackbarPos(name, winname)
            out.append(val)
        return out

    cv.namedWindow(winname)
    trackbars = {
        "cursor 1": [0, 8192],
        "width 1": [0, 1000],
        "cursor 2": [0, 8192],
        "width 2": [0, 1000],
    }
    createTrackbars(winname, trackbars)


    data_folder = "output2"

    hits_file = pth(data_folder, "hits", "0.pkl")
    imgs_folder = pth(data_folder, "imgs")
    if not os.path.exists(imgs_folder):
        os.makedirs(imgs_folder)

    with open(hits_file, "rb") as pickle_in:
        julia_hits = pickle.load(pickle_in)

    # Create a black image, a window
    img = np.zeros(julia_hits.shape, np.uint8)
    cur1, width1, cur2, width2 = 0, 0, 0, 0
    while True:
        tic = time.time()
        red_bgra = neon_effect(
            julia_hits,
            level=cur1,
            width=width1,
            color_bgr=[62, 0, 255],
        )
        blue_bgra = neon_effect(
            julia_hits,
            level=cur2,
            width=width2,
            color_bgr=[255, 0, 62],
        )

        green_bgra = neon_effect(
            julia_hits,
            level=cur2*2,
            width=width1,
            color_bgr=[0, 1, 0],
        )

        julia_bgr = blend([red_bgra, blue_bgra, green_bgra])
        print(int(1000*(time.time()-tic)))
        # julia_bgr = red.astype(float) + blue.astype(float)
        # julia_bgr[julia_bgr > 255] = 255
        # julia_bgr = julia_bgr.astype(np.uint8)


        cv.imshow('image', julia_bgr)
        k = cv.waitKey(10) & 0xFF
        if k == 27:  # esc
            break

        # get current positions of four trackbars
        cur1, width1, cur2, width2 = getTrackbars(winname, trackbars)

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()