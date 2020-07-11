import numpy as np
import time
import utils
from numba import njit, jit


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


@njit
def zoom_space_to_cartesian(x, y, z, cx, cy):
    cart_x = (x-cx)/2**(z-1) + cx
    cart_y = (y-cy)/2**(z-1) + cy
    return cart_x, cart_y


@njit
def apply_rotation(x, y, z, r_mat):
    x1 = x * r_mat[0, 0] + y * r_mat[0, 1] + z * r_mat[0, 2]
    y1 = x * r_mat[1, 0] + y * r_mat[1, 1] + z * r_mat[1, 2]
    z1 = x * r_mat[2, 0] + y * r_mat[2, 1] + z * r_mat[2, 2]
    return x1, y1, z1


@njit
def apply_translation(x, y, z, pos_xyz):
    x1 = pos_xyz[0] + x
    y1 = pos_xyz[1] + y
    z1 = pos_xyz[2] + z
    return x1, y1, z1


@njit
def compute_julia_pixel(x, y, constant_xy):
    hit = 0
    while x ** 2 + y ** 2 < 100 and hit < 1024:
        x0 = x
        y0 = y
        x1 = x0 ** 2 - y0 ** 2 + constant_xy[0]
        y1 = 2 * x0 * y0 + constant_xy[1]
        x = x1
        y = y1
        hit += 1
    return hit

@njit
def compute_mandelbrot_pixel(x, y):
    cx = x
    cy = y
    hit = 0
    while x ** 2 + y ** 2 < 100 and hit < 1024:
        x0 = x
        y0 = y
        x1 = x0 ** 2 - y0 ** 2 + cx
        y1 = 2 * x0 * y0 + cy
        x = x1
        y = y1
        hit += 1
    return hit


@jit(nopython=True, parallel=True)
def juliaset(dim_xy, pos_xyz, r_mat, constant_xy):
    W, H = dim_xy
    size = max(W, H)
    px_size = 2/size

    julia_hits = np.zeros(shape=(H, W), dtype=np.uint16)

    for j in range(H):
        for i in range(W):
            y = -1 + j * px_size
            x = -1 + i * px_size
            z = 0
            x, y, z = apply_rotation(x, y, z, r_mat)
            x, y, z = apply_translation(x, y, z, pos_xyz)
            x, y = zoom_space_to_cartesian(x, y, z, pos_xyz[0], pos_xyz[1])

            julia_hits[j, i] = compute_julia_pixel(x, y, constant_xy)
    return julia_hits


@jit(nopython=True, parallel=True)
def mandelbrotset(dim_xy, pos_xyz, r_mat):
    W, H = dim_xy
    size = max(W, H)
    px_size = 2/size

    mandelbrot_hits = np.zeros(shape=(H, W), dtype=np.uint16)

    for j in range(H):
        for i in range(W):
            y = -1 + j * px_size
            x = -1 + i * px_size
            z = 0
            x, y, z = apply_rotation(x, y, z, r_mat)
            x, y, z = apply_translation(x, y, z, pos_xyz)
            x, y = zoom_space_to_cartesian(x, y, z, pos_xyz[0], pos_xyz[1])

            mandelbrot_hits[j, i] = compute_mandelbrot_pixel(x, y)
    return mandelbrot_hits

def get_initial_values():
    dim_xy = (500, 500)
    pos_xyz = (0, 0, 0)
    r_mat = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    constant_xy = (-0.8372, -0.1939)
    return dim_xy, pos_xyz, r_mat, constant_xy

def main():
    from numba.typed import List

    # juliaset((-1, 0), (-0.8372, -0.1939), (1000, 1000), 0.5)
    dim_xy = (1000, 1000)

    juliaset((1, 1),
             (0, 0, -10),
             np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
             ]),
             (-0.8372, -0.1939))
    totic = 0
    for zoom in range(20):
        pos_xyz = (0, 0, zoom)
        r_mat = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        constant_xy = (-0.8372, -0.1939)

        tic = time.time()
        julia_hits = juliaset(dim_xy, pos_xyz, r_mat, constant_xy)
        tac = time.time()
        totic += tac-tic
        print(zoom, "elapsed time: ", tac - tic, "s")
        julia_hits = 255*(julia_hits.astype(float)+1)/np.max(julia_hits)
        utils.export_to_png(f"zoom/juliaset_{zoom}", julia_hits)
    print("total time: ", totic, "s")


    # jit 18.00 s
    # njit 18.00 s

if __name__ == '__main__':
    main()