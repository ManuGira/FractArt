import cv2 as cv
import numpy as np
import utils
import numba
from scipy import ndimage, signal

def fill_vertices(vertices, mg_x, mg_y):
    H,W = mg_x.shape
    n = 0
    for x in range(W-1):
        for y in range(H-1):
            vertices[n, 0, 0]

            n += 1

def interp(arr, t):
    t = t % 1
    N = len(arr)-1
    r0 = np.array(arr[int(t*N)])
    r1 = np.array(arr[int(t*N)+1])
    res = r0*(1-t) + r1*t
    return res.astype(int).tolist()



@numba.jit
def local_std(img, ksize):
    H, W = img.shape
    d = ksize//2
    out = np.zeros_like(img)
    for y in range(H):
        print(y)
        top = max(0, y-d)
        bottom = min(H-1, y+d)
        for x in range(W):
            left = max(0, x-d)
            right = min(W-1, x+d)
            bloc = img[top:bottom, left:right]
            val = np.std(bloc)
            out[y, x] = val
    return out

# def convolve2D(img, kernel):
#     h, w = kernel.shape
#     H, W = img.shape
#     Ho, Wo = H+2*(h//2), W+2*(w//2)
#
#     out = np.zeros(shape=(Ho, Wo)
#     for yo in Ho:
#         for xo in Wo:



def image_gradient(img):
    blur_size = 19
    blurr_kernel = np.ones(shape=(blur_size, blur_size))/blur_size**2

    dx_kernel = np.array([[-1.0,   1.0]])
    dy_kernel = np.array([[-1.0], [1.0]])

    dx2_kernel = signal.convolve2d(dx_kernel, dx_kernel, boundary='fill', fillvalue=0.0, mode='full')
    dy2_kernel = signal.convolve2d(dy_kernel, dy_kernel, boundary='fill', fillvalue=0.0, mode='full')

    kernel_x = signal.convolve2d(blurr_kernel, dx2_kernel, boundary='fill', fillvalue=0.0, mode='full')
    kernel_y = signal.convolve2d(blurr_kernel, dy2_kernel, boundary='fill', fillvalue=0.0, mode='full')

    print("go")
    outx = signal.convolve2d(img, kernel_x, mode="same")
    print("go2")
    # outy = signal.convolve2d(img, kernel_y, mode="same")
    print("stop")
    return outx #, outy

def move_grid(img, mg_x, mg_y):
    h, w = mg_x.shape
    for j in range(h):
        for i in range(w):
            x = mg_x[j, i]
            y = mg_y[j, i]
            mat = img[y:y+2, x:x+2]
            dx = mat[0, 1] - mat[0, 0]
            dy = mat[1, 0] - mat[0, 0]

            dx = dx/abs(dx) if dx != 0 else 0
            dy = dy/abs(dy) if dy != 0 else 0
            mg_x[j, i] = mg_x[j, i] + dx
            mg_y[j, i] = mg_y[j, i] + dy

def main():
    img0 = cv.imread("../gallery/julia.png")
    img = img0[:, :, 0]
    out = np.zeros_like(img0)
    triangle_size_w = 30
    triangle_size_h = 26
    H, W = img.shape

    grad = image_gradient(img)
    grad = grad-grad.min()
    grad = 255*grad/grad.max()
    utils.export_to_png("gradient", grad)
    exit()
    # locstd = local_std(img, 21)
    # locstd = locstd/locstd.max() * 255
    # locstd = locstd.astype(int)
    # utils.export_to_png("local_std", locstd)

    mg_x, mg_y = np.meshgrid(range(0, W, triangle_size_w), range(0, H, triangle_size_h))
    mg_x[1::2, :] += triangle_size_w//2

    h, w = mg_x.shape
    if h%2 == 0:
        mg_x = mg_x[:-1, :]
        mg_y = mg_y[:-1, :]
        h -= 1
        out = out[:-triangle_size_h,:,:]

    for k in range(3):
        print("go", k)
        move_grid(img, mg_x, mg_y)
    print("go over")

    N = (mg_x.shape[0]-1)*(mg_x.shape[1]-1)*2
    vertices = np.zeros(shape=(N, 3, 2), dtype=np.int32)
    n = 0
    for j in range(1, h-1, 2):
        for i in range(0, w-1):
            # 4 triangles a chaque passage ici

            # triangle 1:
            vertices[n + 0, 0, :] = [mg_x[j  , i  ], mg_y[j  , i  ]]
            vertices[n + 0, 1, :] = [mg_x[j-1, i+1], mg_y[j-1, i+1]]
            vertices[n + 0, 2, :] = [mg_x[j-1, i  ], mg_y[j-1,   i]]

            # triangle 2
            vertices[n + 1, 0, :] = [mg_x[j, i], mg_y[j, i]]
            vertices[n + 1, 1, :] = [mg_x[j, i+1], mg_y[j, i+1]]
            vertices[n + 1, 2, :] = [mg_x[j-1, i+1], mg_y[j-1, i+1]]

            # triangle 3
            vertices[n + 2, 0, :] = [mg_x[j, i], mg_y[j, i]]
            vertices[n + 2, 1, :] = [mg_x[j+1, i+1], mg_y[j+1, i+1]]
            vertices[n + 2, 2, :] = [mg_x[j, i+1], mg_y[j, i+1]]

            # triangle 4
            vertices[n + 3, 0, :] = [mg_x[j, i], mg_y[j, i]]
            vertices[n + 3, 1, :] = [mg_x[j+1, i], mg_y[j+1, i]]
            vertices[n + 3, 2, :] = [mg_x[j+1, i+1], mg_y[j+1, i+1]]
            n += 4

    T = img.max()
    for i in range(0, len(vertices)):
        vert = vertices[i]
        # color = (i%4+1)*64-1
        # color = int(img[vert[0][1], vert[0][0]])
        cx = int(round(sum(vert[:, 0])/3))
        cy = int(round(sum(vert[:, 1])/3))

        t = float(img[cy, cx])/T
        # attenuate contrast
        # t = t**0.5
        # convert t to color
        color = interp([
                [255, 255, 255],
                [0, 0, 0],
            ], t)
        try:
            cv.fillPoly(out, [vert], color)
        except:
            print(i, color)

    utils.export_to_png("triSubSamp", out)


if __name__ == '__main__':
    main()