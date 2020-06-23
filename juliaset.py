import numpy as np
import time
import utils
from numba import jit


@jit
def meshgrid(range_x, range_y):
    # return np.meshgrid(range_x, range_y)
    H = len(range_y)
    W = len(range_x)
    dtype = range_x.dtype
    mgx = np.empty(shape=(H, W), dtype=dtype)
    mgy = np.empty(shape=(H, W), dtype=dtype)
    for j, y in enumerate(range_y):
        mgy[j, :] = y
    # range_y.shape = (H, 1)
    # mgy = np.repeat(range_y, repeats=W, axis=1)
    #
    #
    for i, x in enumerate(range_x):
        mgx[:, i] = x
    return mgx, mgy


@jit
def zoom_space_to_cartesian(x, y, z, cx, cy):
    cart_x = (x-cx)/2**(z-1) + cx
    cart_y = (y-cy)/2**(z-1) + cy
    return cart_x, cart_y


@jit
def compute_julia_numba(mgx, mgy, c, max_iter=80):
    height, width = mgx.shape
    julia_hits = np.zeros(shape=(height, width), dtype=np.uint8)
    for j in range(height):
        rowx = mgx[j]
        rowy = mgy[j]
        for i in range(width):
            x = rowx[i]
            y = rowy[i]
            hit = 0
            while x**2+y**2 < 100 and hit < max_iter:
                x0 = x
                y0 = y
                x1 = x0**2-y0**2 + c[0]
                y1 = 2 * x0 * y0 + c[1]
                x = x1
                y = y1
                hit += 1
            julia_hits[j, i] = hit
    return julia_hits


def compute_julia(mgx, mgy, c, max_iter=80):
    height, width = mgx.shape
    cc = complex(c[0], c[1])
    mgc = mgx + mgy * 1j
    julia_hits = np.zeros(shape=(height, width), dtype=np.uint8)
    mask = abs(mgc) < 10
    for k in range(max_iter):
        mgc[mask] = mgc[mask] ** 2 + cc
        mask = abs(mgc) < 10
        julia_hits[mask] = julia_hits[mask] + 1
    return julia_hits


@jit
def generate_mesh(width, height, ratio, cx, cy, zoom):
    range_x = cx + np.linspace(-width / 2, width / 2, width) * ratio
    range_y = cy + np.linspace(-height / 2, height / 2, height) * ratio
    mgx, mgy = meshgrid(range_x, range_y)
    mgz = np.zeros_like(mgx) + zoom
    for j in range(height):
        for i in range(width):
            x, y = zoom_space_to_cartesian(mgx[j, i], mgy[j, i], mgz[j, i], cx, cy)
            mgx[j, i] = x
            mgy[j, i] = y
    return mgx, mgy

@jit
def generate_mesh2(dim_xy, pos_xyz, r_mat):
    W, H = dim_xy
    ratio = max(W, H)
    range_x = np.linspace(-W, W, W) / ratio
    range_y = np.linspace(-H, H, H) / ratio
    xs, ys = meshgrid(range_x, range_y)
    zs = np.zeros_like(xs)

    mgx = np.zeros_like(xs)
    mgy = np.zeros_like(xs)

    # TODO: rotate and translate xs, ys, zs, to mgx, mgy, mgz
    for j in range(H):
        for i in range(W):
            x0 = xs[j, i]
            y0 = ys[j, i]
            z0 = zs[j, i]
            x1 = pos_xyz[0] + x0*r_mat[0, 0] + y0*r_mat[0, 1] + z0*r_mat[0, 2]
            y1 = pos_xyz[1] + x0*r_mat[1, 0] + y0*r_mat[1, 1] + z0*r_mat[1, 2]
            z1 = pos_xyz[2] + x0*r_mat[2, 0] + y0*r_mat[2, 1] + z0*r_mat[2, 2]

            x, y = zoom_space_to_cartesian(x1, y1, z1, pos_xyz[0], pos_xyz[1])
            mgx[j, i] = x
            mgy[j, i] = y
    return mgx, mgy


@jit
def juliaset2(dim_xy, pos_xyz, r_mat, constant_xy):
    mgx, mgy = generate_mesh2(dim_xy, pos_xyz, r_mat)
    julia_hits = compute_julia_numba(mgx, mgy, constant_xy, 255)
    return julia_hits


def juliaset(center_xy, constant_xy, dim_xy, zoom):
    cx, cy = center_xy
    pos_xyz = cx, cy, 1
    width, height = dim_xy
    ratio = 2/max(width, height)/zoom
    # print(f"{width}x{height} = {width*height} pixels")
    print(width, "x", height, " = ", width*height, "pixels")

    for zoom in range(20):
        tic = time.time()
        julia_hits = juliaset2(width, height, ratio, cx, cy, zoom, constant_xy)
        utils.export_to_png(f"zoom/julia_{zoom}", julia_hits)
        print("elapsed time: ", time.time() - tic, "s")


def main():
    from numba.typed import List

    # juliaset((-1, 0), (-0.8372, -0.1939), (1000, 1000), 0.5)
    dim_xy = (1000, 1000)

    for zoom in range(5):
        pos_xyz = (0, 0, zoom)
        r_mat = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        constant_xy = (-0.8372, -0.1939)

        tic = time.time()
        julia_hits = juliaset2(dim_xy, pos_xyz, r_mat, constant_xy)
        print("elapsed time: ", time.time() - tic, "s")
        utils.export_to_png(f"zoom/juliaset_{zoom}", julia_hits)


if __name__ == '__main__':
    main()