import numpy as np
import time
import utils
from numba import njit, jit, vectorize


@njit
def zoom_space_to_cartesian(x, y, z, cx, cy):
    cart_x = (x-cx)/2**(z-1) + cx
    cart_y = (y-cy)/2**(z-1) + cy
    return cart_x, cart_y


@njit
def apply_fisheye(x, y, z, fisheye_factor):
    x1 = x
    y1 = y
    z1 = z + (x*x + y*y)*fisheye_factor
    return x1, y1, z1

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
def compute_mandelbrot_pixel(x, y, max_iter):
    cx = x
    cy = y
    hit = 0
    while x ** 2 + y ** 2 < 100 and hit < max_iter:
        x0 = x
        y0 = y
        x1 = x0 ** 2 - y0 ** 2 + cx
        y1 = 2 * x0 * y0 + cy
        x = x1
        y = y1
        hit += 1
    return hit

# @njit
# def screen_space_to_zoom_space(dim_xy, pos_xy, zoom, r_mat, pos_screen_xy):
#     W, H = dim_xy
#     pos_xyz = pos_xy + (zoom,)
#     size = max(W, H)
#     px_size = 2 / size
#
#     i, j = pos_screen_xy
#
#     y = -1 + j * px_size
#     x = -1 + i * px_size
#     z = 0
#     x, y, z = apply_rotation(x, y, z, r_mat)
#     x, y, z = apply_translation(x, y, z, pos_xyz)
#     return x, y, z

def cartesian_space_to_screen(dim_xy, pos_xy, zoom, r_mat, pos_cart_xy):
    """
    TODO:
    interection de la droite du point pos_cart_xy dans le zoom space avec le screen.
    la droite passe part les 2 points
        (pos_cartxy[0], pos_cartxy[0], 0)
    et
        (2*pos_cartxy[0], 2*pos_cartxy[0], 1)

    le plan du screen passe part le point:
        (pos_xy[0], pos_xy[1], zoom)
    et a comme vecteur normal:
        r_mat [:, 2]

    :param dim_xy:
    :param pos_xy:
    :param zoom:
    :param r_mat:
    :param pos_cart_xy:
    :return:
    """



@njit
def screen_space_to_cartesian(dim_xy, pos_xy, zoom, r_mat, pos_screen_xy, fisheye_factor=0):
    W, H = dim_xy
    pos_xyz = pos_xy + (zoom,)
    size = max(W, H)
    px_size = 2 / size

    i, j = pos_screen_xy

    y = -1 + j * px_size
    x = -1 + i * px_size
    z = 0
    x, y, z = apply_fisheye(x, y, z, fisheye_factor)
    x, y, z = apply_rotation(x, y, z, r_mat)
    x, y, z = apply_translation(x, y, z, pos_xyz)
    x, y = zoom_space_to_cartesian(x, y, z, pos_xyz[0], pos_xyz[1])
    return x, y


@njit()
def juliaset_none(dim_xy, pos_xy, zoom, r_mat, constant_xy, supersampling=1, fisheye_factor=0, max_iter=1024):
    """
    this is a fake julia function, just to measure the cpu time needed for computing the meshgrids
    :return: None
    """
    W, H = dim_xy
    pos_xyz = pos_xy + (zoom,)
    size = max(W, H)
    px_size = 2 / size
    sub_px_size = px_size / supersampling

    dx = min(1.0, W / H)
    dy = min(1.0, H / W)
    julia_hits = np.zeros(shape=(H, W), dtype=np.int32) + max_iter
    for super_j in range(supersampling):
        for super_i in range(supersampling):
            mesh_x = np.zeros(shape=(H, W), dtype=np.float64)
            mesh_y = np.zeros(shape=(H, W), dtype=np.float64)
            for j in range(H):
                # print(100*j/H, "%")
                for i in range(W):
                    y = -dy + j * px_size + super_j * sub_px_size
                    x = -dx + i * px_size + super_i * sub_px_size
                    z = 0
                    x, y, z = apply_fisheye(x, y, z, fisheye_factor)
                    x, y, z = apply_rotation(x, y, z, r_mat)
                    x, y, z = apply_translation(x, y, z, pos_xyz)
                    x, y = zoom_space_to_cartesian(x, y, z, pos_xyz[0], pos_xyz[1])
                    mesh_x[j, i] = x
                    mesh_y[j, i] = y
    return julia_hits


def compute_julia_numpy(mgx, mgy, c, max_iter):
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


def juliaset_numpy(dim_xy, pos_xy, zoom, r_mat, constant_xy, supersampling=1, fisheye_factor=0, max_iter=1024):
    W, H = dim_xy
    pos_xyz = pos_xy + (zoom,)
    size = max(W, H)
    px_size = 2 / size
    sub_px_size = px_size / supersampling

    dx = min(1.0, W / H)
    dy = min(1.0, H / W)
    julia_hits = np.zeros(shape=(H, W), dtype=np.int32) + max_iter
    for super_j in range(supersampling):
        for super_i in range(supersampling):
            mesh_x = np.zeros(shape=(H, W), dtype=np.float64)
            mesh_y = np.zeros(shape=(H, W), dtype=np.float64)
            for j in range(H):
                # print(100*j/H, "%")
                for i in range(W):
                    y = -dy + j * px_size + super_j * sub_px_size
                    x = -dx + i * px_size + super_i * sub_px_size
                    z = 0
                    x, y, z = apply_fisheye(x, y, z, fisheye_factor)
                    x, y, z = apply_rotation(x, y, z, r_mat)
                    x, y, z = apply_translation(x, y, z, pos_xyz)
                    x, y = zoom_space_to_cartesian(x, y, z, pos_xyz[0], pos_xyz[1])
                    mesh_x[j, i] = x
                    mesh_y[j, i] = y
            hits = compute_julia_numpy(mesh_x, mesh_y, constant_xy, max_iter)
            julia_hits = np.min((julia_hits, hits), axis=0)
    return julia_hits

@njit
def compute_julia_pixel(x, y, constant_xy, max_iter):
    hit = 0
    while x ** 2 + y ** 2 < 100 and hit < max_iter:
        x0 = x
        y0 = y
        x1 = x0 ** 2 - y0 ** 2 + constant_xy[0]
        y1 = 2 * x0 * y0 + constant_xy[1]
        x = x1
        y = y1
        hit += 1
    return hit


@jit(nopython=True, parallel=True)
def juliaset_njit(dim_xy, pos_xy, zoom, r_mat, constant_xy, supersampling=1, fisheye_factor=0, max_iter=1024):
    W, H = dim_xy
    pos_xyz = pos_xy + (zoom,)
    size = max(W, H)
    px_size = 2/size
    sub_px_size = px_size / supersampling

    dx = min(1.0, W/H)
    dy = min(1.0, H/W)

    julia_hits = np.zeros(shape=(H, W), dtype=np.uint16)
    for j in range(H):
        # print(100*j/H, "%")
        for i in range(W):
            min_hits = max_iter
            for super_j in range(supersampling):
                for super_i in range(supersampling):
                    y = -dy + j * px_size + super_j*sub_px_size
                    x = -dx + i * px_size + super_i*sub_px_size
                    z = 0
                    x, y, z = apply_fisheye(x, y, z, fisheye_factor)
                    x, y, z = apply_rotation(x, y, z, r_mat)
                    x, y, z = apply_translation(x, y, z, pos_xyz)
                    x, y = zoom_space_to_cartesian(x, y, z, pos_xyz[0], pos_xyz[1])

                    hits = compute_julia_pixel(x, y, constant_xy, max_iter=min_hits)
                    min_hits = min(min_hits, hits)
            julia_hits[j, i] = min_hits
    return julia_hits


@vectorize(['int64(float64, float64, float64, float64, int64)'], target="cpu")
def compute_julia_pixel_vectorized(x, y, constant_x, constant_y, max_iter):
    hit = 0
    while x ** 2 + y ** 2 < 100 and hit < max_iter:
        x0 = x
        y0 = y
        x1 = x0 ** 2 - y0 ** 2 + constant_x
        y1 = 2 * x0 * y0 + constant_y
        x = x1
        y = y1
        hit += 1
    return hit


@jit(nopython=True, parallel=True)
def juliaset_vectorized(dim_xy, pos_xy, zoom, r_mat, constant_xy, supersampling=1, fisheye_factor=0, max_iter=1024):
    W, H = dim_xy
    pos_xyz = pos_xy + (zoom,)
    size = max(W, H)
    px_size = 2 / size
    sub_px_size = px_size / supersampling

    dx = min(1.0, W / H)
    dy = min(1.0, H / W)
    julia_hits = np.zeros(shape=(H, W), dtype=np.int32) + max_iter
    for super_j in range(supersampling):
        for super_i in range(supersampling):
            mesh_x = np.zeros(shape=(H, W), dtype=np.float64)
            mesh_y = np.zeros(shape=(H, W), dtype=np.float64)
            for j in range(H):
                # print(100*j/H, "%")
                for i in range(W):
                    y = -dy + j * px_size + super_j * sub_px_size
                    x = -dx + i * px_size + super_i * sub_px_size
                    z = 0
                    x, y, z = apply_fisheye(x, y, z, fisheye_factor)
                    x, y, z = apply_rotation(x, y, z, r_mat)
                    x, y, z = apply_translation(x, y, z, pos_xyz)
                    x, y = zoom_space_to_cartesian(x, y, z, pos_xyz[0], pos_xyz[1])
                    mesh_x[j, i] = x
                    mesh_y[j, i] = y
            julia_hits = compute_julia_pixel_vectorized(mesh_x, mesh_y, constant_xy[0], constant_xy[1], julia_hits)
    return julia_hits


@jit(nopython=True, parallel=True)
def mandelbrotset(dim_xy, pos_xy, zoom, r_mat, max_iter=1024):
    W, H = dim_xy
    size = max(W, H)
    px_size = 2/size

    pos_xyz = pos_xy + (zoom,)

    mandelbrot_hits = np.zeros(shape=(H, W), dtype=np.uint16)

    for j in range(H):
        for i in range(W):
            y = -1 + j * px_size
            x = -1 + i * px_size
            z = 0
            x, y, z = apply_rotation(x, y, z, r_mat)
            x, y, z = apply_translation(x, y, z, pos_xyz)
            x, y = zoom_space_to_cartesian(x, y, z, pos_xyz[0], pos_xyz[1])

            mandelbrot_hits[j, i] = compute_mandelbrot_pixel(x, y, max_iter)
    return mandelbrot_hits

def get_initial_values():
    dim_xy = (500, 500)
    pos_julia_xy = (0, 0)
    zoom = 0
    r_mat = np.eye(3)
    constant_xy = (-0.8372, -0.1939)  # constant is the position xy of mandelbrot
    return dim_xy, pos_julia_xy, zoom, r_mat, constant_xy

def main():
    # juliaset((-1, 0), (-0.8372, -0.1939), (1000, 1000), 0.5)
    dim_xy = (1000, 1000)

    juliaset_njit((1, 1),
                  (0, 0, -10),
                  np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
             ]),
                  (-0.8372, -0.1939))
    totic = 0
    for zoom in range(20):
        pos_julia_xy = (0, 0)
        r_mat = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        constant_xy = (-0.8372, -0.1939)

        tic = time.time()
        julia_hits = juliaset_njit(dim_xy, pos_julia_xy, zoom, r_mat, constant_xy)
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