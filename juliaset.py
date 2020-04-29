import numpy as np
import cv2 as cv
import time
import os

def export_to_png(filename, data):
    folder = "gallery"
    if not os.path.exists(folder):
        os.mkdir(folder)
    file = os.path.join(folder, f"{filename}.png")
    cv.imwrite(file, data)

def juliaset(center_xy, dim_xy, zoom):
    cx, cy = center_xy
    width, height = dim_xy
    ratio = 2/max(width, height)/zoom

    range_x = cx + np.linspace(-width/2, width/2, width)*ratio
    range_y = cy + np.linspace(-height/2, height/2, height)*ratio
    mgx, mgy = np.meshgrid(range_x, range_y)
    mgz = mgx + mgy * 1j

    c = complex(-0.8372, -0.1939)
    julia_hits = np.zeros(shape=(height, width), dtype=np.uint8)

    tic = time.time()
    mask = abs(mgz) < 10
    for k in range(255):
        print(f"{100*k/255: 3.2f}%")
        mgz[mask] = mgz[mask]**2 + c
        mask = abs(mgz) < 10
        julia_hits[mask] = julia_hits[mask] + 1

        cv.imshow("julia", julia_hits)
        key = cv.waitKey(1)
    print(time.time()-tic)

    cv.imshow("julia", julia_hits)
    key = cv.waitKey(0)

    export_to_png("julia", julia_hits)


if __name__ == '__main__':
    juliaset((0,0),(200,160), 0.75)
