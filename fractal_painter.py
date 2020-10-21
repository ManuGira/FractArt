import numpy as np
import numba
import cv2 as cv

class FractalPainter:
    def __init__(self, max_iter, colorbar_path=None, texture_path=None):
        self.max_iter = max_iter
        if colorbar_path is not None:
            self.colorbar = load_colorbar(colorbar_path)
        if texture_path is not None:
            self.texture = load_texture(texture_path)

    def paint_standard(self, julia_hits, hits_offset=0, gradient_factor=0, use_glow_effect=False, use_fake_supersampling=False):
        """
        Standard painting using an hardcoded colorbar
        """
        if use_fake_supersampling:
            julia_hits = fake_supersampling(julia_hits)

        if hits_offset != 0:
            julia_hits += hits_offset
            julia_hits[julia_hits > self.max_iter - 1] = self.max_iter - 1
        julia_bgr = color_map(julia_hits, self.max_iter)

        if gradient_factor != 0:
            julia_bgr = add_gradient(julia_bgr, julia_hits, gradient_factor)

        if use_glow_effect:
            julia_bgr = glow_effect(julia_bgr)
        return julia_bgr

    def paint_colorbar(self, julia_hits, hits_offset=0, gradient_factor=0, use_glow_effect=False, use_fake_supersampling=False):
        """
        Use the colorbar img passed in constructor (loaded with load_colorbar())
        """
        if use_fake_supersampling:
            julia_hits = fake_supersampling(julia_hits)

        if hits_offset != 0:
            julia_hits += hits_offset
            julia_hits[julia_hits > self.max_iter - 1] = self.max_iter - 1
        julia_bgr = apply_color_map(julia_hits, self.colorbar)

        if gradient_factor != 0:
            julia_bgr = add_gradient(julia_bgr, julia_hits, gradient_factor)

        if use_glow_effect:
            julia_bgr = glow_effect(julia_bgr)
        return julia_bgr

    def paint_texture(self, julia_hits, julia_trap_magn, julia_trap_phase, hits_offset=0, gradient_factor=0, use_glow_effect=False, use_fake_supersampling=False):
        """
        Use the texture passed in costructor (loaded with load_texture())
        """
        if use_fake_supersampling:
            julia_hits = fake_supersampling(julia_hits)

        julia_hits += hits_offset
        julia_hits[julia_hits > self.max_iter - 1] = self.max_iter - 1
        phase01 = julia_trap_phase / (2 * np.pi) + 0.5
        magn01 = julia_trap_magn * np.log(julia_hits + 1)

        julia_bgr = texture_map(phase01, magn01, self.texture)

        if gradient_factor != 0:
            julia_bgr = add_gradient(julia_bgr, julia_hits, gradient_factor)

        if use_glow_effect:
            julia_bgr = glow_effect(julia_bgr)

        return julia_bgr

    def paint_colorbar_as_texture(self, julia_hits, julia_trap_magn, julia_trap_phase, hits_offset=0, gradient_factor=0, use_glow_effect=False, use_fake_supersampling=False):
        """
        Paint using paint_colorbar(), and use the result as a texture for paint_texture
        """
        texture = self.paint_colorbar(
            julia_hits,
            hits_offset=hits_offset,
            gradient_factor=0,
            use_glow_effect=use_glow_effect,
            use_fake_supersampling=use_fake_supersampling,
        )
        self.texture = format_texture(texture)
        julia_bgr = self.paint_texture(
            julia_hits,
            julia_trap_magn,
            julia_trap_phase,
            hits_offset=0,
            gradient_factor=gradient_factor,
            use_glow_effect=use_glow_effect,
            use_fake_supersampling=use_fake_supersampling,
        )
        return julia_bgr




@numba.njit
def apply_color_map(hits, color_map):
    """
    :param hits: np array of int values
    :param color_map: (N x 1 x 3) np array
    :return: painted img
    """

    H, W = hits.shape
    N, _, _ = color_map.shape
    out = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    for j in range(H):
        for i in range(W):
            k = hits[j, i] % N
            bgr = color_map[k]
            out[j, i, :] = bgr
    return out


def color_gradient(colors_bgr, length, interpolation=cv.INTER_LINEAR):
    colors_bgr = np.array(colors_bgr, dtype=np.uint8)
    return cv.resize(colors_bgr, dsize=(3, length), interpolation=interpolation)


def color_gradient_2(colors_bgr0, colors_bgr1, length):
    if length < 0 :
        colors_bgr0, colors_bgr1 = colors_bgr1[:], colors_bgr0[:]
        length = abs(length)
    b0, g0, r0 = colors_bgr0
    b1, g1, r1 = colors_bgr1

    out = np.zeros((length, 3), dtype=np.uint8)

    out[:, 0] = (np.linspace(b0, b1, length)+0.5).astype(np.uint8)
    out[:, 1] = (np.linspace(g0, g1, length)+0.5).astype(np.uint8)
    out[:, 2] = (np.linspace(r0, r1, length)+0.5).astype(np.uint8)
    return out


def color_map(hits, max_iter):
    # dB = lambda x: (20 * np.log(x + 1)).astype(np.uint16)
    # hits = dB(hits)
    # BRG colors
    N = 8192
    color_list = [
        [255, 255, 255],
        [0, 0, 0],
        [255, 255, 95],
        [0, 0, 0],
        [127, 127, 255],
        [0, 0, 0],
        [127, 255, 255],
        [0, 0, 0],
    ]*16
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


def load_colorbar(path):
    """
    load a colorbar from an image (it takes the first line of pixel of the image)
    :param path: path to colorbar image
    :return: (Nx3) np array
    """
    colorbar = cv.imread(path)
    colorbar = colorbar[0:1, :, :]
    # turn colorbar shape from (1, N, 3) to (N, 1, 3)
    colorbar = np.reshape(colorbar, newshape=(-1, 1, 3))
    return colorbar


def format_texture(texture):
    H, W, _ = texture.shape
    mx = max(H, W)
    return cv.resize(texture, (mx, mx))


def load_texture(path):
    texture = cv.imread(path)
    return format_texture(texture)


def texture_map(mesh_x, mesh_y, texture):
    H, W = texture.shape[:2]
    mesh_u = (np.mod(mesh_x, 1)*W).astype(np.float32)
    mesh_v = (np.mod(mesh_y, 1)*H).astype(np.float32)
    out = cv.remap(texture, mesh_u, mesh_v, cv.INTER_LINEAR)
    return out


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


# @vectorize(['(float64, float64, float64)(float64, float64, float64)'], target="cpu")
def cvtBRG_to_HLScube_vectorized(b, g, r):
    mx = max(b, g, r)
    mn = min(b, g, r)
    mxn = mx+mn

    l = mxn/2

    if (mx == mn):
        return 0, l, 0

    dmxn = mx-mn
    s = dmxn/(2-mxn) if mxn > 1 else dmxn/mxn

    if mx == b:
        h = (r - g) / dmxn + 4
    elif mx == g:
        h = (b - r) / dmxn + 2
    else:
        h = (g - b) / dmxn
        h = h+6 if g < b else h
    h /= 6
    return h, l, s


def cvtHLScube_to_BGR(hls):
    hls[:, :, 0] *= 180
    hls[:, :, 1] *= 255
    hls[:, :, 2] *= 255
    hls = hls.astype(np.uint8)
    bgr = cv.cvtColor(hls, cv.COLOR_HLS2BGR)
    return bgr


def neon_effect2(hits, colors, brigther_factor=2, glow_size1=3):
    # TODO: make it numba
    H, W, = hits.shape
    mask_bgr = np.zeros(shape=(H, W, 3), dtype=np.uint16)

    for color in colors:
        color_bgr = color['bgr']
        level = color['level']
        width = color['width']

        hls = cv.cvtColor(np.array([[color_bgr]], dtype=np.uint8), cv.COLOR_BGR2HLS)  # uint8 (1x1x3)
        hls = hls.squeeze()

        mask_gray = np.zeros(shape=(H, W), dtype=np.uint8)
        mask = np.logical_and(level - width < hits, hits < level + width)
        mask_gray[mask] = hls[1]
        mask_bgr[:, :, 0][mask] += color_bgr[0]
        mask_bgr[:, :, 1][mask] += color_bgr[1]
        mask_bgr[:, :, 2][mask] += color_bgr[2]

    mask_bgr[mask_bgr > 255] = 255
    mask_bgr = mask_bgr.astype(np.uint8)
    mask_hls = cv.cvtColor(mask_bgr, cv.COLOR_BGR2HLS)

    light = mask_hls[:, :, 2].astype(np.uint16)*brigther_factor
    glow = mask_bgr.copy()
    mask_bgr = mask_bgr.astype(np.uint16)
    K = 10
    for k in range(K):
        glow = cv.GaussianBlur(glow, ksize=(0, 0), sigmaX=glow_size1, sigmaY=0)
        mask_bgr += glow
        light += cv.cvtColor(glow, cv.COLOR_BGR2GRAY)
    mask_bgr = mask_bgr / (K+1)
    mask_hls = cv.cvtColor(mask_bgr.astype(np.uint8), cv.COLOR_BGR2HLS)

    light = light // brigther_factor
    light[light > 255] = 255
    light = light.astype(np.uint8)

    # bgr to hls, shape (1,1,3)
    mask_hls[:, :, 1] = light

    light.shape += (1,)
    mask_bgra = np.concatenate((
        cv.cvtColor(mask_hls, cv.COLOR_HLS2BGR),
        light,
    ), axis=2)
    return mask_bgra


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


# @numba.vectorize('uint8(uint8, uint8)', locals={'out': numba.uint8}, target="cpu")
# @numba.stencil('uint8(uint8, uint8)')
@numba.vectorize('uint8(uint8, uint8)', target="parallel")
def add_saturate_uint8(bgr0, bgr1):
    out = bgr0 + bgr1
    # out[out < bgr0] = 255
    if out > 255:
        out = 255
    return out


def add_gradient(bgr, hits, gradient_factor):
    # compute gradient
    gradient_xy = compute_gradient(hits)

    # merge gradient to bgr
    hls = cvtBRG_to_HLScube(bgr)
    hls[:, :, 1] = hls[:, :, 1] + gradient_xy[:, :, 0] * gradient_factor
    hls[hls > 1] = 1
    hls[hls < 0] = 0
    bgr = cvtHLScube_to_BGR(hls)
    return bgr

def compute_gradient(hits, normalizing_threshold=64):
    """

    :param hits: (M x N) np array of integer
    :param normalizing_threshold: normalizing and clipping threshold. Lower threshold will increase gradient effects.
    :return: (M x N) np array of type float64
    """
    H, W = hits.shape
    gradient_xy = np.zeros(shape=(H, W, 2), dtype=hits.dtype)
    gradient_xy[:, :, 0] = cv.filter2D(hits.astype(float), cv.CV_64F, np.array([[-1, 1]]))
    gradient_xy[:, :, 1] = cv.filter2D(hits.astype(float), cv.CV_64F, np.array([-1, 1]))

    # compute gradient magnitude
    grad_magn = np.sqrt(gradient_xy[:, :, 0] ** 2 + gradient_xy[:, :, 1] ** 2)
    grad_magn.shape += (1,)
    grad_magn = np.repeat(grad_magn, repeats=2, axis=2)

    # clip gradient magnitude to threshold
    mask = grad_magn > normalizing_threshold
    gradient_xy[mask] = normalizing_threshold * gradient_xy[mask] / grad_magn[mask]
    gradient_xy = gradient_xy.astype(np.float64) / normalizing_threshold

    return gradient_xy


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

@numba.njit
def fake_supersampling(julia_hits):
    out = np.zeros_like(julia_hits)
    H, W = out.shape
    for j in range(1, H-1):
        for i in range(1, W-1):
            out[j, i] = np.min(julia_hits[j-1:j+1, i-1:i+1])
    return out

    # cv.morphologyEx(
    #         src=julia_hits,
    #         op=cv.MORPH_ERODE,
    #         kernel=cv.getStructuringElement(),
    #     )

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
        colors = [
            {"bgr": [0, 0, 255], 'level': cur1, "width": width1},
            {"bgr": [0, 255, 255], 'level': cur2, "width": width2},
        ]
        red_bgra = neon_effect2(
            julia_hits,
            colors,
        )
        # blue_bgra = neon_effect(
        #     julia_hits,
        #     level=cur2,
        #     width=width2,
        #     color_bgr=[255, 0, 62],
        # )
        #
        # green_bgra = neon_effect(
        #     julia_hits,
        #     level=cur2*2,
        #     width=width1,
        #     color_bgr=[0, 1, 0],
        # )

        # julia_bgr = blend([red_bgra, blue_bgra, green_bgra])
        # print(int(1000*(time.time()-tic)))
        # julia_bgr = red.astype(float) + blue.astype(float)
        # julia_bgr[julia_bgr > 255] = 255
        # julia_bgr = julia_bgr.astype(np.uint8)


        cv.imshow('image', red_bgra[:,:])
        k = cv.waitKey(10) & 0xFF
        if k == 27:  # esc
            break

        # get current positions of four trackbars
        cur1, width1, cur2, width2 = getTrackbars(winname, trackbars)

    cv.destroyAllWindows()

def main2():
    import time
    img0 = cv.imread('outputs/output_posters/imgs/7.png', cv.IMREAD_GRAYSCALE)
    img1 = cv.imread('outputs/output_posters/imgs/9.png', cv.IMREAD_GRAYSCALE)

    # img0 = np.array([
    #     [100, 100],
    #     [100, 100]
    # ], dtype=np.uint8)

    # img1 = np.array([
    #     [1, 200],
    #     [255, 0]
    # ], dtype=np.uint8)

    img = add_saturate_uint8(img0, img1)
    tic = time.time()
    for k in range(100):
        img = add_saturate_uint8(img0, img1)
    toc = time.time()

    print(toc-tic)

    print(img)

    # cv.imshow("img0", img0)
    # cv.imshow("img1", img1)
    # cv.imshow("img", img)

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main2()