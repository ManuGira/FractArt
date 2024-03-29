import numpy as np
import time
import utils
from numba import njit, jit, vectorize, guvectorize, cuda
import math


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
        mask = abs(mgc) < 2
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
    x2 = x * x
    y2 = y * y
    while x2 + y2 < 4 and hit < max_iter:
        y = (x+x) * y + constant_xy[1]
        x = x2 - y2 + constant_xy[0]
        y2 = y*y
        x2 = x*x
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


# @vectorize(['int64(float64, float64, float64, float64, int64)'], target="cuda")
@vectorize(['int64(float64, float64, float64, float64, int64)'], target="cpu")
def compute_julia_pixel_vectorized(x, y, constant_x, constant_y, max_iter):
    hit = 0
    x2 = x * x
    y2 = y * y
    while x2 + y2 < 4 and hit < max_iter:
        y = (x+x) * y + constant_y
        x = x2 - y2 + constant_x
        y2 = y*y
        x2 = x*x
        hit += 1
    return hit

@cuda.jit
def compute_julia_pixel_cuda_kernel(mesh_x, mesh_y, constant_x, constant_y, max_iter, out_hit):
    i, j = cuda.grid(2)  # +
    H, W = mesh_x.shape  # +
    if 0 <= i < W and 0 <= j < H:   # +
        x = mesh_x[j, i]  # +
        y = mesh_y[j, i]  # +
        mxi = max_iter[j, i]  # +
        hit = 0
        x2 = x * x
        y2 = y * y
        while x2 + y2 < 4 and hit < mxi:
            y = (x+x) * y + constant_y
            x = x2 - y2 + constant_x
            y2 = y*y
            x2 = x*x
            hit += 1
        out_hit[j, i] = hit  # +


@guvectorize(
    ["f8[:], f8[:], f8, f8, i8, i8[:], f8[:], f8[:]"],
    '(n),(n),(),(),()->(n),(n),(n)',
    target='parallel')
def compute_julia_traps_pixel_guvectorized(
        x, y, constant_x, constant_y, max_iter,  # inputs
        out_hit, out_dist, out_theta  # outputs
):
    for i in range(x.shape[0]):
        x_i = x[i]
        y_i = y[i]

        hit = 0
        x2 = x_i * x_i
        y2 = y_i * y_i
        dist2 = x2 + y2
        min_dist = dist2
        min_dist_x = y_i
        min_dist_y = x_i
        while dist2 < 4 and hit < max_iter:
            y_i = (x_i+x_i) * y_i + constant_y
            x_i = x2 - y2 + constant_x
            y2 = y_i*y_i
            x2 = x_i*x_i
            dist2 = x2 + y2
            if dist2 < min_dist:
                min_dist = dist2
                min_dist_x = x_i
                min_dist_y = y_i
            hit += 1
        out_hit[i] = hit
        out_dist[i] = (min_dist**0.5)
        out_theta[i] = np.arctan2(min_dist_x, min_dist_y)


def juliaset_trapped_guvectorized(dim_xy, pos_xy, zoom, r_mat, constant_xy, supersampling=1, fisheye_factor=0, max_iter=1024):
    """
    compute juliaset with orbital traps
    """
    W, H = dim_xy
    pos_xyz = pos_xy + (zoom,)
    size = max(W, H)
    px_size = 2 / size
    sub_px_size = px_size / supersampling

    constant_xy = [float(c) for c in constant_xy]

    dx = min(1.0, W / H)
    dy = min(1.0, H / W)
    julia_hits = np.zeros(shape=(H, W), dtype=np.int64) + max_iter
    julia_trap_magn = np.zeros(shape=(H, W), dtype=np.float64) + 2
    julia_trap_phase = np.zeros(shape=(H, W), dtype=np.float64)

    hits = np.zeros(shape=(H, W), dtype=np.int64) + max_iter
    trap_magn = np.zeros(shape=(H, W), dtype=np.float64) + 2
    trap_phase = np.zeros(shape=(H, W), dtype=np.float64)

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

            compute_julia_traps_pixel_guvectorized(mesh_x, mesh_y, constant_xy[0], constant_xy[1], max_iter, hits, trap_magn, trap_phase)
            updated_hits_mask = hits < julia_hits
            julia_trap_magn[updated_hits_mask] = trap_magn[updated_hits_mask]
            julia_trap_phase[updated_hits_mask] = trap_phase[updated_hits_mask]
            julia_hits[updated_hits_mask] = hits[updated_hits_mask]
    return julia_hits, julia_trap_magn, julia_trap_phase


# comment decorator for cuda
@jit(parallel=True, target='cpu')
def juliaset_vectorized(dim_xy, pos_xy, zoom, r_mat, constant_xy, supersampling=1, fisheye_factor=0, max_iter=1024):
    W, H = dim_xy
    pos_xyz = pos_xy + (zoom,)
    size = max(W, H)
    px_size = 2 / size
    sub_px_size = px_size / supersampling

    constant_xy = [float(c) for c in constant_xy]

    dx = min(1.0, W / H)
    dy = min(1.0, H / W)
    julia_hits = np.zeros(shape=(H, W), dtype=np.int64) + max_iter
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


def juliaset_cuda_kernel(dim_xy, pos_xy, zoom, r_mat, constant_xy, supersampling=1, fisheye_factor=0, max_iter=1024):
    # TODO: this is not a trap version. Must implement orbital trap
    W, H = dim_xy
    pos_xyz = pos_xy + (zoom,)
    size = max(W, H)
    px_size = 2 / size
    sub_px_size = px_size / supersampling

    constant_xy = [float(c) for c in constant_xy]

    threads_per_block = (32, 32)
    blocks_per_grid_x = math.ceil(W / threads_per_block[0])
    blocks_per_grid_y = math.ceil(H / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    dx = min(1.0, W / H)
    dy = min(1.0, H / W)
    julia_hits = np.zeros(shape=(H, W), dtype=np.int64) + max_iter
    hits = np.zeros(shape=(H, W), dtype=np.int64)
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
            compute_julia_pixel_cuda_kernel[blocks_per_grid, threads_per_block](mesh_x, mesh_y, constant_xy[0], constant_xy[1], julia_hits, hits)
            julia_hits = hits.copy()
    julia_trap_magn = np.zeros(shape=(H, W), dtype=np.float64)
    julia_trap_phase = np.zeros(shape=(H, W), dtype=np.float64)
    return julia_hits, julia_trap_magn, julia_trap_phase


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