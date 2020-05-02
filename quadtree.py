import cv2 as cv
import numpy as np
import utils


class QuadTree:

    def __init__(self, N, img, rectangle=None):
        self.N = N
        self.img = img
        # coordinates of the rectangle points in the original image
        if rectangle is None:
            # p0 and p1 in xy order [px0 py0] [px1 py1]
            self.rectangle = [0, 0], [self.img.shape[1], self.img.shape[0]]
        else:
            self.rectangle = rectangle

        self.childs = []
        if N > 0 and self.cond_func():
            self.compute_child_nodes()

    def get_rectangles(self):
        out = []
        for child in self.childs:
            out += child.get_rectangles()
        if len(out) == 0:  # we are a leave
            return [self.rectangle]
        return out

    def get_Ns(self):
        out = []
        for child in self.childs:
            out += child.get_Ns()
        if len(out) == 0:  # we are a leave
            return [self.N]
        return out

    def compute_child_nodes(self):
        H, W = self.img.shape[0:2]
        midH = int(round(H/2))
        midW = int(round(W/2))
        Nc = self.N-1

        px0, py0 = self.rectangle[0][0], self.rectangle[0][1]
        rect00 = [px0 +    0, py0 +    0], [px0 + midW, py0 + midH]
        rect01 = [px0 + midW, py0 +    0], [px0 +    W, py0 + midH]
        rect10 = [px0 +    0, py0 + midH], [px0 + midW, py0 +    H]
        rect11 = [px0 + midW, py0 + midH], [px0 +    W, py0 +    H]

        child00 = QuadTree(Nc, self.img[:midH, :midW], rect00)
        child01 = QuadTree(Nc, self.img[:midH, midW:], rect01)
        child10 = QuadTree(Nc, self.img[midH:, :midW], rect10)
        child11 = QuadTree(Nc, self.img[midH:, midW:], rect11)
        self.childs = [child00, child01, child10, child11]

    def cond_func(self):
        return self.img.max() > 100


def main():
    img0 = cv.imread("gallery/bigjulia4.png")
    img = img0[:, :, 0]
    N = 12
    root = QuadTree(N, img)
    rects = root.get_rectangles()
    Ns = root.get_Ns()
    print(len(rects))

    pts0 = lambda r: (r[0][0], r[0][1])
    pts1 = lambda r: (r[1][0], r[0][1])
    pts2 = lambda r: (r[1][0], r[1][1])
    pts3 = lambda r: (r[0][0], r[1][1])
    rects = [[pts0(r), pts1(r), pts2(r), pts3(r)] for r in rects]
    rects = np.array(rects, np.int32)

    img0 = np.zeros_like(img0)
    for i in range(len(rects)):
        color = int(255*((Ns[i])/(N-2))**0.5)
        # cv.polylines(img0, [rect], True, (np.random.randint(0, 255), 255, 255))
        img0 = cv.fillPoly(img0, [rects[i]], (color,color,color))

    # cv.imshow("OK", img0)
    # cv.waitKey(0)
    utils.export_to_png("quad", img0)

if __name__ == '__main__':
    main()