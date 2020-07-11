import cv2 as cv
import numpy as np
import juliaset

class FractalExplorer:

    def __init__(self):
        self.julia_hits = None
        self.mandel_hits = None
        self.liveImg = None
        self.mousePosition = (0, 0)
        self.winName = 'image'
        self.info = ""

        dim_xy, pos_xyz, r_mat, constant_xy = juliaset.get_initial_values()
        self.dim_xy = dim_xy
        self.pos_xyz = pos_xyz
        self.r_mat = r_mat
        self.constant_xy = constant_xy


    def update_julia_hits(self):
        self.julia_hits = juliaset.juliaset(self.dim_xy, self.pos_xyz, self.r_mat, self.constant_xy)


    def update_mandel_hits(self):
        self.mandel_hits = juliaset.mandelbrotset(self.dim_xy, self.pos_xyz, self.r_mat)


    def display(self):
        self.info = f"pos: {self.pos_xyz}, C: {self.constant_xy}"
        julia_display = (255*(self.julia_hits.astype(float)+1)/np.max(self.julia_hits)).astype(np.uint8)
        mandel_display = (255*(self.mandel_hits.astype(float)+1)/np.max(self.mandel_hits)).astype(np.uint8)
        self.liveImg = np.hstack([julia_display, mandel_display])
        # cv.putText(self.liveImg, info, (20, H-20), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv.LINE_AA)
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
        cv.namedWindow(self.winName)
        cv.setMouseCallback(self.winName, self.mouseCallback)

        self.update_julia_hits()
        self.update_mandel_hits()
        key = '-'
        while key not in ' ':
            self.display()
            key = chr(cv.waitKey(0) & 0xFF)

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

