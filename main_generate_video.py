import pickle
import juliaset
import fractal_painter
from utils import pth
import utils
import os
import cv2 as cv
import time
import numpy as np



def generate_video_from_folder(data_folder, fps):
    print("generate_video_from_folder: ", end="")
    tic = time.time()

    imgs_folder = pth(data_folder, "imgs")
    video_path = pth(data_folder, "juliaset.mp4")
    img0 = cv.imread(pth(imgs_folder, "0.png"))
    H, W, _ = img0.shape

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv.VideoWriter(video_path, fourcc, fps, (W, H))

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

    for keyword in ["zoom", "r_mat", "fisheye_factor"]:
        out[keyword] = locA[keyword] * (1 - t) + t * locB[keyword]

    return out


def estimate_computation_time(itinary, dim_xy, nb_inter_frame, supersampling):
    time_per_pixels = [location["time_per_px"] for location in itinary]
    time_per_pixels[0] *= 0.5
    time_per_pixels[-1] *= 0.5
    avg_time_per_pixels = sum(time_per_pixels)/(len(time_per_pixels)-1)
    W, H = dim_xy
    out = avg_time_per_pixels * W*H * nb_inter_frame * (len(itinary)-1) * supersampling**2
    # no idea why, but it works better by mutliplying by 0.7
    return 0.7*out


def generate_hits_from_itinary(data_folder, dim_xy, nb_inter_frame, supersampling):
    print("generate_hits_from_itinary")
    tic0 = time.time()

    with open(pth(data_folder, "itinary.pkl"), "rb") as pickle_in:
        itinary = pickle.load(pickle_in)

    print(f"Itinary made of {len(itinary)} locations interpolated by {nb_inter_frame} frames -> total = {(len(itinary)-1)*nb_inter_frame} frames")
    estimated_time = estimate_computation_time(itinary, dim_xy, nb_inter_frame, supersampling)
    eh, em, es = utils.sec_to_hms(estimated_time)
    print(f"estimated computation time: {eh}h {em}m {es}s")

    hits_folder = pth(data_folder, "hits")
    if not os.path.exists(hits_folder):
        os.makedirs(hits_folder)

    k = 0
    for i in range(len(itinary)-1):
        print(f"{i} ", end="")
        tic1 = time.time()
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
                supersampling=supersampling,
                fisheye_factor=location["fisheye_factor"],
            )

            with open(pth(hits_folder, f"{k}.pkl"), "wb") as pickle_out:
                pickle.dump(julia_hits, pickle_out)
            k += 1
            print('.', end='')
        print(f" {time.time()-tic1:.4f}s")
    print(f"Total time: {time.time()-tic0:.1f}s")



if __name__ == '__main__':
    data_folder = "output"
    dim_xy = (720, 540)
    nb_inter_frame = 60
    supersampling = 3
    fps = 60

    generate_hits_from_itinary(data_folder, dim_xy, nb_inter_frame, supersampling)
    generate_images_from_hits(data_folder)
    generate_video_from_folder(data_folder, fps)
