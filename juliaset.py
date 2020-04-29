import numpy as np
import cv2 as cv
import time

def juliaset(center_xy, dim_xy, zoom):
    cx, cy = center_xy
    width, height = dim_xy
    ratio = 2/max(width, height)/zoom

    range_x = cx + np.linspace(-width/2, width/2, width)*ratio
    range_y = cy + np.linspace(-height/2, height/2, height)*ratio
    mgx, mgy = np.meshgrid(range_x, range_y)
    print(mgx)
    print(mgy)
    mgz = mgx + mgy * 1j

    c = complex(-0.8372, -0.1939)
    julia_hits = np.zeros(shape=(height, width), dtype=np.uint8)

    tic = time.time()
    for i in range(height):
        for j in range(width):
            z = mgz[i, j]
            count = 0
            while abs(z) < 10 and count < 255:
                z = z**2 + c
                count += 1
            hits[i, j] = count
    print(time.time()-tic)
    cv.imshow("julia", hits)
    key = cv.waitKey(0)



if __name__ == '__main__':
    juliaset((0,0),(200,160), 0.75)
