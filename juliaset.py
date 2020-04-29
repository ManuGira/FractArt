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
    mask = abs(mgz) < 10
    for k in range(255):
        print(k)
        mgz[mask] = mgz[mask]**2 + c
        mask = abs(mgz) < 10
        hits[mask] = hits[mask] + 1
    print(time.time()-tic)

    
    cv.imshow("julia", hits)
    key = cv.waitKey(0)



if __name__ == '__main__':
    juliaset((0,0),(200,160), 0.75)
