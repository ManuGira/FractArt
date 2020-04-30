import cv2 as cv


class QuadTree:
    def __init__(self, img, N, condition_function):
        self.img = img
        self.imgfb = img.copy()
        self.N = N

        self.dim = img.shape[0:2]
        self.root = QuadTreeNode(0, 0, self.dim[0], self.dim[1])
        self.cond_func = condition_function
        self.stages = [[self.root]] + [[]]*self.N

    def compute(self):
        for n in range(self.N):
            print(f"Stage {n}, {len(self.stages[n])} nodes")
            nodes = self.stages[n] + []
            for node in nodes:

                node.compute_child_nodes(self.img, self.cond_func)
                if len(node.childs) > 0:
                    self.stages[n+1] += node.childs

                    self.imgfb = cv.rectangle(self.imgfb, (node.left+2, node.top+2), (node.right-2, node.bottom-2), color=(255,))
                    cv.imshow("OK", self.imgfb)
                    cv.waitKey(1)
        print(self.stages)

class QuadTreeNode:

    def __init__(self, top, left, bottom, right):
        self.top = top
        self.left = left
        self.bottom = bottom
        self.right = right
        self.childs = []

    def compute_child_nodes(self, img, condition_function):
        midx = int(round(self.right+self.left)/2)
        midy = int(round(self.bottom+self.top)/2)

        child00 = QuadTreeNode(self.top, self.left, midy, midx)
        child01 = QuadTreeNode(self.top, midx, midy, self.right)
        child10 = QuadTreeNode(midy, self.left, self.bottom, midx)
        child11 = QuadTreeNode(midy, midx, self.bottom, self.right)
        for child in [child00, child01, child10, child11]:
            if condition_function(img, child):
                self.childs.append(child)

def condition_function(img, node):
    subimg = img[node.top:node.bottom, node.left:node.right]
    print("\t", node.top, node.left, node.bottom, node.right)
    try:
        subimg.max()
    except:
        print("oh non")
    print("\t\t", subimg.max())
    return subimg.max() > 25

def main():
    img = cv.imread("gallery/julia.png")
    img = img[:, :, 0]
    qt = QuadTree(img, 10, condition_function)
    qt.compute()
    print("ok")



if __name__ == '__main__':
    main()