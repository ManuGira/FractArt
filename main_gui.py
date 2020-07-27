import cv2 as cv
import numpy as np
import juliaset
import time
import pickle
import fractal_painter
from utils import pth
# from threading import Thread

# cv_process = Thread(target=pss.run, kwargs={"online": False})
# cv_process.start()


class FractalExplorer:
    def __init__(self, data_folder):
        self.julia_hits = None
        self.mandel_hits = None
        self.julia_display = None
        self.mandel_display = None
        self.liveImg = None
        self.pos_mouse_julia_screen_xy = (0, 0)
        self.winName = 'image'
        self.info = ""

        dim_xy, pos_julia_xy, zoom, r_mat, constant_xy = juliaset.get_initial_values()
        self.dim_xy = dim_xy
        W, H = dim_xy
        self.pos_julia_xy = pos_julia_xy
        self.zoom_julia = zoom
        self.delta_zoom_mandel = 5
        self.r_mat = r_mat
        self.pos_mandel_xy = constant_xy
        self.fisheye_factor = 0.0

        self.pos_mouse_julia_xy = 0, 0
        self.pos_mouse_mandel_xy = 0, 0
        self.pos_mouse_julia_screen_xy = W//2, H//2
        self.light_effect = -1

        self.itinary_path = pth(data_folder, "itinary.pkl")
        self.itinary = []
        self.time_per_px = None

        # run juliaset function once to compile them
        juliaset.juliaset((1, 1), self.pos_julia_xy, self.zoom_julia, self.r_mat, self.pos_mandel_xy)

    def update_julia_hits(self):
        tic = time.time()
        self.julia_hits = juliaset.juliaset(
            self.dim_xy,
            self.pos_julia_xy,
            self.zoom_julia,
            self.r_mat,
            self.pos_mandel_xy,
            fisheye_factor=self.fisheye_factor,
        )
        W, H = self.dim_xy
        self.time_per_px = (time.time()-tic) / (W*H)

    def update_mandel_hits(self):
        self.mandel_hits = juliaset.mandelbrotset(
            self.dim_xy,
            self.pos_mandel_xy,
            self.zoom_julia + self.delta_zoom_mandel,
            self.r_mat,
        )


    def update_julia_display(self):
        self.julia_display = fractal_painter.color_map(self.julia_hits)
        self.julia_display = fractal_painter.glow_effect(self.julia_display)
        H, W = self.julia_display.shape[:2]
        cv.line(self.julia_display, (W // 2, 48 * H // 100), (W // 2, 52 * H // 100), (0, 127, 127))
        cv.line(self.julia_display, (48 * W // 100, H // 2), (52 * W // 100, H // 2), (0, 127, 127))

    def update_mandel_display(self):
        self.mandel_display = fractal_painter.color_map(self.mandel_hits)
        H, W = self.mandel_display.shape[:2]
        cv.line(self.mandel_display, (W//2, 48*H//100), (W//2, 52*H//100), (0, 127, 127))
        cv.line(self.mandel_display, (48*W//100, H//2), (52*W//100, H//2), (0, 127, 127))


    def putText(self):
        self.info = f"""Julia pos: {self.pos_julia_xy}
        Mandel pos: {self.pos_mandel_xy}
        Zoom julia: 2^{self.zoom_julia:.2f}
        julia mouse:  ({self.pos_mouse_julia_xy[0]:.3f}, {self.pos_mouse_julia_xy[1]:.3f})
        mandel mouse: ({self.pos_mouse_mandel_xy[0]:.3f}, {self.pos_mouse_mandel_xy[1]:.3f})"""
        H = self.liveImg.shape[0]
        infos = self.info.split("\n        ")
        pos_y = H-20*len(infos)
        for k, txt in enumerate(infos):
            cv.putText(self.liveImg, txt, (20, pos_y+20*k), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 2, cv.LINE_AA)
            cv.putText(self.liveImg, txt, (20, pos_y+20*k), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1, cv.LINE_AA)


    def display(self):
        self.liveImg = np.hstack([self.julia_display, self.mandel_display])
        self.putText()
        cv.imshow(self.winName, self.liveImg)


    def mouseCallback(self, event, x, y, flags, param):
        W, H = self.dim_xy

        if x < W:
            # mouse on juliaset
            self.pos_mouse_julia_screen_xy = x, y
            self.pos_mouse_julia_xy = juliaset.screen_space_to_cartesian(
                self.dim_xy,
                self.pos_julia_xy,
                self.zoom_julia,
                self.r_mat,
                self.pos_mouse_julia_screen_xy,
                self.fisheye_factor,
            )

            if event == cv.EVENT_LBUTTONUP:
                self.pos_julia_xy = self.pos_mouse_julia_xy
                self.update_julia_hits()
                self.update_mandel_hits()
                self.update_julia_display()
                self.update_mandel_display()

            # self.light_effect = x/W
        else:
            # mouse on mandelbrot
            pos_mouse_mandel_screen_xy = x - W, y
            self.pos_mouse_mandel_xy = juliaset.screen_space_to_cartesian(
                self.dim_xy, self.pos_mandel_xy, self.zoom_julia + self.delta_zoom_mandel, self.r_mat, pos_mouse_mandel_screen_xy)
            if event == cv.EVENT_LBUTTONUP:
                print(x, y)
                self.pos_mandel_xy = self.pos_mouse_mandel_xy
                self.update_mandel_hits()
                self.update_mandel_display()
                self.update_julia_hits()
                self.update_julia_display()
        self.display()


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

            if key in 's':
                # saving current location to itinary
                location = {
                    "pos_julia_xy": self.pos_julia_xy,
                    "zoom": self.zoom_julia,
                    "r_mat": self.r_mat,
                    "pos_mandel_xy": self.pos_mandel_xy,
                    "fisheye_factor": self.fisheye_factor,
                    "time_per_px": self.time_per_px,
                }
                self.itinary.append(location)
                print(self.itinary)
                with open(self.itinary_path, "wb") as pickle_out:
                    pickle.dump(self.itinary, pickle_out)

            if key in 'x':
                self.fisheye_factor = (self.fisheye_factor+0.1+1)%2-1
                self.update_julia_hits()
                self.update_julia_display()

            if key in 'z':
                self.zoom_julia += 1
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


def main(data_folder):
    fe = FractalExplorer(data_folder)
    fe.start()


def printPressedKey():
    cv.namedWindow('Pressed Key')
    k = 0
    while k != 27:
        k = cv.waitKey(0) & 0xFF
        print(k)

if __name__ == '__main__':
    # printPressedKey()
    data_folder = "output"
    main(data_folder)

