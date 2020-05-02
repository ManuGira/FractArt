import numpy as np
import cv2 as cv
import time
import os
import utils


def juliaset(center_xy, dim_xy, zoom):
    cx, cy = center_xy
    width, height = dim_xy
    ratio = 2/max(width, height)/zoom
    print(f"{width}x{height} = {width*height} pixels")

    range_x = cx + np.linspace(-width/2, width/2, width)*ratio
    range_y = cy + np.linspace(-height/2, height/2, height)*ratio
    mgx, mgy = np.meshgrid(range_x, range_y)
    mgz = mgx + mgy * 1j

    c = complex(-0.8372, -0.1939)
    julia_hits = np.zeros(shape=(height, width), dtype=np.uint8)

    tic = time.time()
    mask = abs(mgz) < 10
    for k in range(255):
        print(f"{100*k/255: 5.2f}%")
        mgz[mask] = mgz[mask]**2 + c
        mask = abs(mgz) < 10
        julia_hits[mask] = julia_hits[mask] + 1

    print(f"elapsed time: {time.time()-tic:.3f}s")
    utils.export_to_png("julia", julia_hits)


if __name__ == '__main__':
    juliaset((0,0),(708, 472), 0.5)
