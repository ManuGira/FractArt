import cv2 as cv
import numpy as np
import utils
import math


class QuadTree:

    def __init__(self, N, img, rectangle=None):
        maxN = int(math.log2(min(img.shape[0:2])))
        if N > maxN:
            print(f"Max N is {maxN} for image with dimension {img.shape}")
            N = maxN
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
            pts0 = lambda r: (r[0][0], r[0][1])
            pts1 = lambda r: (r[1][0], r[0][1])
            pts2 = lambda r: (r[1][0], r[1][1])
            pts3 = lambda r: (r[0][0], r[1][1])
            r = self.rectangle
            rectangle = [pts0(r), pts1(r), pts2(r), pts3(r)]
            return [rectangle]
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


class QuadTreeForce:
    k = 0
    K = 1
    rootN = 1
    def __init__(self, N, img, origin_xy=None, triangle=None):
        maxN = int(math.log2(min(img.shape[0:2])))
        if N > maxN:
            print(f"Max N is {maxN} for image with dimension {img.shape}")
            N = maxN
        self.N = N
        self.img = img
        # The origin are the XY coordinates of the top-left corner of the current img in the original image
        if origin_xy is None:
            self.origin_xy = np.array([0, 0], dtype=np.int)
            QuadTreeForce.K = 4**N
            QuadTreeForce.rootN = N
        else:
            self.origin_xy = origin_xy

        if triangle is None:
            # p0 and p1 in xy order [px0 py0] [px1 py1]
            H, W = self.img.shape[0:2]
            p0xy = np.array([W//2, int(H*0.8)], dtype=np.int)
            p1xy = np.array([W//2, int(H*0.2)], dtype=np.int)
            p2xy = QuadTreeForce.line2triangle(p0xy, p1xy)
            self.triangle = np.array([p0xy, p1xy, p2xy])
        else:
            self.triangle = triangle

        self.childs = []
        if N > 0 and self.cond_func():
            self.compute_child_nodes()
        else:
            QuadTreeForce.k += 4**N
            if QuadTreeForce.rootN-N < 5:
                print(f"{100*QuadTreeForce.k/QuadTreeForce.K:.2f}%")

    def compute_child_nodes(self):
        Nc = self.N - 1

        p0, p1, p2 = self.triangle
        p01 = (p0+p1)//2
        p12 = (p1+p2)//2
        p20 = (p2+p0)//2

        triangle0 = np.array([p0, p01, p20])
        triangle1 = np.array([p1, p12, p01])
        triangle2 = np.array([p2, p20, p12])
        triangle3 = np.array([p01, p12, p20])

        for triangle in [triangle0, triangle1, triangle2, triangle3]:
            # range of the triangles in order [[minX, minY], [maxX, maxY]] (in local coordinates)
            span = [np.min(triangle, axis=0) - self.origin_xy, np.max(triangle, axis=0) - self.origin_xy]
            subimg = self.img[span[0][1]:span[1][1], span[0][0]:span[1][0]]
            origin = np.min(triangle, axis=0)
            child = QuadTreeForce(Nc, subimg, origin, triangle)
            self.childs.append(child)

    def cond_func(self):
        mask = np.array(self.img > 100).reshape((-1,))
        if not any(mask):
            return False

        # convert triangle to local coordinates
        triangle_local = self.triangle-self.origin_xy
        v01xy = triangle_local[1]-triangle_local[0]
        v12xy = triangle_local[2]-triangle_local[1]
        v20xy = triangle_local[0]-triangle_local[2]
        H, W = self.img.shape[0:2]
        mg_x, mg_y = np.meshgrid(range(W), range(H))
        pixels_xy = np.concatenate((mg_x.reshape((-1, 1)), mg_y.reshape((-1, 1))), axis=1)
        for i in range(3):
            # vector from point to pixel
            point = triangle_local[i]
            vector = [v01xy, v12xy, v20xy][i]
            vp = pixels_xy - point
            # if a pixel is outside the triangle, then the cross product will be negative
            cross_positive = vector[1]*vp[:, 0] > vector[0]*vp[:, 1]
            mask = np.logical_and(mask, cross_positive)

            # toshow = utils.draw_poly(mask, triangle_local, reshape=(H,W))
            # utils.imshow(toshow)

            if not any(mask):
                return False
        return True

    def get_triangles(self):
        out = []
        for child in self.childs:
            out += child.get_triangles()
        if len(out) == 0:  # we are a leave
            return [self.triangle]
        return out

    def get_Ns(self):
        out = []
        for child in self.childs:
            out += child.get_Ns()
        if len(out) == 0:  # we are a leave
            return [self.N]
        return out

    @staticmethod
    def line2triangle(p0xy, p1xy):
        """
        Given the first 2 points of a triangle, returns the third point
         - p0xy and p1xy must be numpy arrays of shapes (2,) with dtype=np.int
         - coordinates comes in xy order
         - returned p2xy satisfy the condition (p1xy-p0xy)x(p2xy-p0xy) > 0 (cross product)
         - triangle is equilateral
        """
        v01xy = p1xy-p0xy
        v01xy_perp = np.array([v01xy[1], -v01xy[0]])
        h = np.cos(np.pi/6)
        v02xy = 0.5*v01xy + h*v01xy_perp
        p2xy = p0xy + v02xy
        return np.array(np.round(p2xy), dtype=np.int)

def main():
    img0 = cv.imread("../gallery/bigjulia4.png")
    img = img0[:, :, 0]
    N = 7
    root = QuadTreeForce(N, img)
    pts = root.get_triangles()
    Ns = root.get_Ns()
    print(len(pts))

    pts = np.array(pts, np.int32)

    img0 = np.ones_like(img0)*255
    x = 50
    for i in range(len(pts)):
        color = int((255-x)*((Ns[i])/(N-2))**0.5)
        # cv.polylines(img0, [rect], True, (np.random.randint(0, 255), 255, 255))
        img0 = cv.fillPoly(img0, [pts[i]], (np.random.randint(0, x)+color,color,color))
    img1 = img0[::-1,::-1,:]
    img1[img1>img0] = img0[img1>img0]

    # cv.imshow("OK", img0)
    # cv.waitKey(0)
    utils.export_to_png("quad", img0)

if __name__ == '__main__':
    main()