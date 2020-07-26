import pickle
import juliaset
import fractal_painter
from utils import pth
import os
import cv2 as cv
import time



def generate_video_from_folder(data_folder):
    print("generate_video_from_folder: ", end="")
    tic = time.time()

    imgs_folder = pth(data_folder, "imgs")
    video_path = pth(data_folder, "juliaset.mp4")
    img0 = cv.imread(pth(imgs_folder, "0.png"))
    H, W, _ = img0.shape

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv.VideoWriter(video_path, fourcc, 60.0, (W, H))

    K = len(os.listdir(imgs_folder))
    for k in range(K):
        image_path = pth(imgs_folder, f"{k}.png")
        frame = cv.imread(image_path)

        out.write(frame)  # Write out frame to video

        if ((100 * k / K) // 10) < ((100 * (k + 1) / K) // 10):
            print("X", end="")
    print(f" {time.time()-tic:.4f}s")
    # Release everything if job is finished
    out.release()


def generate_images_from_hits(data_folder):
    print("generate_images_from_hits: ", end="")
    tic = time.time()


    hits_folder = pth(data_folder, "hits")
    imgs_folder = pth(data_folder, "imgs")
    if not os.path.exists(imgs_folder):
        os.makedirs(imgs_folder)

    K = len(os.listdir(hits_folder))
    for k in range(K):
        with open(pth(hits_folder, f"{k}.pkl"), "rb") as pickle_in:
            julia_hits = pickle.load(pickle_in)
        julia_bgr = fractal_painter.color_map(julia_hits)
        julia_bgr = fractal_painter.glow_effect(julia_bgr)
        cv.imwrite(pth(imgs_folder, f"{k}.png"), julia_bgr)
        if ((100*k/K)//10) < ((100*(k+1)/K)//10):
            print("X", end="")
    print(f" {time.time()-tic:.4f}s")


def interpolate_locations(locA, locB, t):
    out = {}
    for keyword in ["pos_julia_xy", "pos_mandel_xy"]:
        x = locA[keyword][0] * (1 - t) + t * locB[keyword][0]
        y = locA[keyword][1] * (1 - t) + t * locB[keyword][1]
        out[keyword] = x, y

    for keyword in ["zoom", "r_mat"]:
        out[keyword] = locA[keyword] * (1 - t) + t * locB[keyword]

    return out


def generate_hits_from_itinary(data_folder):
    print("generate_hits_from_itinary")
    tic = time.time()

    nb_inter_frame = 60
    dim_xy = (720, 540)

    with open(pth(data_folder, "itinary.pkl"), "rb") as pickle_in:
        itinary = pickle.load(pickle_in)


    print(f"Itinary made of {len(itinary)} locations interpolated by {nb_inter_frame} frames -> total = {(len(itinary)-1)*nb_inter_frame} frames")

    hits_folder = pth(data_folder, "hits")
    if not os.path.exists(hits_folder):
        os.makedirs(hits_folder)

    k = 0
    for i in range(len(itinary)-1):
        print(f"{i} ", end="")
        tic = time.time()
        locA = itinary[i]
        locB = itinary[i+1]
        for j in range(nb_inter_frame):
            t = j/nb_inter_frame
            location = interpolate_locations(locA, locB, t)
            julia_hits = juliaset.juliaset(
                dim_xy,
                location["pos_julia_xy"],
                location["zoom"],
                location["r_mat"],
                location["pos_mandel_xy"],
                supersampling=3)

            with open(pth(hits_folder, f"{k}.pkl"), "wb") as pickle_out:
                pickle.dump(julia_hits, pickle_out)
            k += 1
            print('.', end='')
        print(" ")
    print(f" {time.time()-tic:.4f}s")



if __name__ == '__main__':
    data_folder = "output"
    generate_hits_from_itinary(data_folder)
    generate_images_from_hits(data_folder)
    generate_video_from_folder(data_folder)
