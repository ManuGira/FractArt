import cv2 as cv
import numpy as np
import juliaset
import time


class FractalExplorer:
    def __init__(self):
        self.julia_hits = None
        self.mandel_hits = None
        self.julia_display = None
        self.mandel_display = None
        self.liveImg = None
        self.mousePosition = (0, 0)
        self.winName = 'image'
        self.info = ""

        dim_xy, pos_julia_xy, zoom, r_mat, constant_xy = juliaset.get_initial_values()
        self.dim_xy = dim_xy
        self.pos_julia_xy = pos_julia_xy
        self.zoom = zoom
        self.r_mat = r_mat
        self.pos_mandel_xy = constant_xy

    def update_julia_hits(self):
        self.julia_hits = juliaset.juliaset(self.dim_xy, self.pos_julia_xy, self.zoom, self.r_mat, self.pos_mandel_xy)

    def update_mandel_hits(self):
        self.mandel_hits = juliaset.mandelbrotset(self.dim_xy, self.pos_mandel_xy, self.zoom, self.r_mat)

    def color_map(self, hits):
        maxhit = np.max(hits)
        return (hits.astype(float) * 255/maxhit).astype(np.uint8)

    def update_julia_display(self):
        self.julia_display = self.color_map(self.julia_hits)

    def update_mandel_display(self):
        self.mandel_display = self.color_map(self.mandel_hits)
        # TODO: convert constant_xy to center_xy
        cv.circle(self.mandel_display, (200, 200), 3, (127,))

    def putText(self):
        self.info = f"Julia pos: {self.pos_julia_xy}\nMandel pos: {self.pos_mandel_xy}\nZoom: 2^{self.zoom:.2f}"
        H = self.liveImg.shape[0]
        infos = self.info.split("\n")
        pos_y = H-20*len(infos)
        for k, txt in enumerate(infos):
            cv.putText(self.liveImg, txt, (20, pos_y+20*k), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv.LINE_AA)


    def display(self):
        self.liveImg = np.hstack([self.julia_display, self.mandel_display])
        self.putText()
        cv.imshow(self.winName, self.liveImg)


    def mouseCallback(self, event, x, y, flags, param):
        # print(event, flags, param)
        if event == cv.EVENT_LBUTTONUP:
            pass

        # mouse position in pixels on image 0
        self.mousePosition = x, y
        # self.display()


    def start(self):
        print(f"Start fractal explorer: ")

        self.update_julia_hits()
        self.update_mandel_hits()
        self.update_julia_display()
        self.update_mandel_display()

        cv.namedWindow(self.winName)
        cv.setMouseCallback(self.winName, self.mouseCallback)

        key = '-'
        while key not in ' ':
            self.display()
            key = chr(cv.waitKey(0) & 0xFF)

            if key in 'u':
                print("updating...")
                tic = time.time()
                self.update_julia_hits()
                self.update_mandel_hits()
                self.update_julia_display()
                self.update_mandel_display()
                dt = time.time() - tic
                print(f"computed {2*self.dim_xy[0]*self.dim_xy[1]} pixels in {dt:.3f} s")

            if key in 'q':
                print("closing...")
                return

        print("terminated")


def main():
    fe = FractalExplorer()
    fe.start()


def printPressedKey():
    cv.namedWindow('Pressed Key')
    k = 0
    while k != 27:
        k = cv.waitKey(0) & 0xFF
        print(k)

if __name__ == '__main__':
    # printPressedKey()
    main()
