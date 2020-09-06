import pickle
import juliaset
import fractal_painter
from utils import pth
import utils
import os
import cv2 as cv
import time
import numpy as np
import scipy.io.wavfile
import sound_itinary_planner


def progression_bar(k, K):
    modA = 10
    modB = 100

    modA, modB = min(modA, modB), max(modA, modB)

    # if ((100 * k / K) // 10) < ((100 * (k + 1) / K) // 10):
    modX = lambda x, X, m: (m*x) // X

    condA = modX(k, K, modA) < modX(k+1, K, modA)
    condB = modX(k, K, modB) < modX(k+1, K, modB)
    if condA:
        print("X", end="", flush=True)
    elif condB:
        print(".", end="", flush=True)

def generate_video_from_folder(data_folder, fps):
    if fps <= 0:
        print("fps must be a positive number")
    print("generate_video_from_folder: ")
    tic = time.time()

    imgs_folder = pth(data_folder, "imgs")
    video_path = pth(data_folder, "juliaset.mp4")
    img0 = cv.imread(pth(imgs_folder, "0.png"))
    H, W, _ = img0.shape

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv.VideoWriter(video_path, fourcc, fps, (W, H))

    K = len(os.listdir(imgs_folder))
    frame_prev = img0*0
    t = 5
    for k in range(K):
        image_path = pth(imgs_folder, f"{k}.png")

        frame = cv.imread(image_path)

        # video post processing
        frame = frame-(frame//t) + (frame_prev//t)
        # frame = cv.addWeighted(frame, t, frame_prev, 1-t, 0)
        frame_prev = frame.copy()

        out.write(frame)  # Write out frame to video

        progression_bar(k, K)
    print(f" {time.time()-tic:.4f}s")
    # Release everything if job is finished
    out.release()


def generate_images_from_hits(data_folder, full_itinary, fps):
    PROFILER=False

    print("generate_images_from_hits: ", end="")
    tic = time.time()

    hits_folder = pth(data_folder, "hits")
    imgs_folder = pth(data_folder, "imgs")
    if not os.path.exists(imgs_folder):
        os.makedirs(imgs_folder)

    K = len(os.listdir(hits_folder))
    print(f"{K} hits matrices to paint")
    for k in range(K):
        with open(pth(hits_folder, f"{k}.pkl"), "rb") as pickle_in:
            julia_hits = pickle.load(pickle_in)

        tic1 = time.time()
        julia_hits = fractal_painter.fake_supersampling(julia_hits)
        print(f"fake sampling: {time.time()-tic1:.4f}s", end="") if PROFILER else None
        # julia_bgr = fractal_painter.color_map(julia_hits, max_iter)
        # julia_bgr = fractal_painter.glow_effect(julia_bgr)

        loc = full_itinary[k]
        width = 30 + 3*loc["sidechains"]["GHOST"]["volume"]

        # TODO: trigger a new color on each GHOST.wav hit.

        # quadratic scale multiplier 1 < f(k) < 4
        quadratic_scale = (((k%1024)/1024)+1)**2
        red_position = (k*100/fps)%1024
        blue_position = (k*200/fps+512)%1024

        colors = [
            {"bgr": [127, 0, 255], 'level': red_position*quadratic_scale, "width": width*quadratic_scale},
            {"bgr": [255, 0, 64], 'level': blue_position*quadratic_scale, "width": width*quadratic_scale},
        ]
        tic2 = time.time()
        julia_bgr = fractal_painter.neon_effect2(julia_hits, colors, brigther_factor=quadratic_scale)
        print(f", neon_effect2: {time.time()-tic2:.4f}s", end="") if PROFILER else None

        tic3 = time.time()
        cv.imwrite(pth(imgs_folder, f"{k}.png"), julia_bgr)
        print(f", imwrite: {time.time()-tic3:.4f}s", flush=True) if PROFILER else None

        if ((100*k/K)//10) < ((100*(k+1)/K)//10):
            print("X", end="")
    print(f" {time.time()-tic:.4f}s")


def estimate_computation_time(itinary, dim_xy, nb_inter_frame, supersampling):
    time_per_pixels = [location["time_per_px"] for location in itinary]
    time_per_pixels[0] *= 0.5
    time_per_pixels[-1] *= 0.5
    avg_time_per_pixels = sum(time_per_pixels)/(len(time_per_pixels)-1)
    W, H = dim_xy
    out = avg_time_per_pixels * W*H * nb_inter_frame * (len(itinary)-1) * supersampling**2
    return out


def generate_hits_from_itinary(data_folder, dim_xy, full_itinary, supersampling, max_iter):
    # run juliaset function once to compile it
    # juliaset.juliaset_njit((1, 1), (0, 0), 1, np.eye(3), (0, 0))
    juliaset.juliaset_vectorized((1, 1), (0, 0), 1, np.eye(3), (0, 0))

    print("generate_hits_from_itinary")
    tic0 = time.time()

    # estimated_time = estimate_computation_time(itinary, dim_xy, nb_inter_frame, supersampling)
    # eh, em, es = utils.sec_to_hms(estimated_time)
    # print(f"estimated computation time: {eh}h {em}m {es}s")

    hits_folder = pth(data_folder, "hits")
    if not os.path.exists(hits_folder):
        os.makedirs(hits_folder)

    k = 0
    N = len(full_itinary)
    for i in range(N - 1):
        print(f"{i}/{N}: ", end="")
        tic1 = time.time()
        loc = full_itinary[i]
        if False:
            julia_hits = juliaset.juliaset_njit(
                dim_xy,
                loc["pos_julia_xy"],
                loc["zoom"],
                loc["r_mat"],
                loc["pos_mandel_xy"],
                supersampling=supersampling,
                fisheye_factor=loc["fisheye_factor"],
                max_iter=max_iter,
            )
        else:
            julia_hits = juliaset.juliaset_vectorized(
                dim_xy,
                loc["pos_julia_xy"],
                loc["zoom"],
                loc["r_mat"],
                loc["pos_mandel_xy"],
                supersampling=supersampling,
                fisheye_factor=loc["fisheye_factor"],
                max_iter=max_iter,
            )
        with open(pth(hits_folder, f"{k}.pkl"), "wb") as pickle_out:
            pickle.dump(julia_hits, pickle_out)
        k += 1
        print(f" {time.time()-tic1:.4f}s", flush=True)
    print(f"Total time: {time.time()-tic0:.1f}s")

def main():
    data_folder = "output3"

    MODE = ["sketchy", "video LD", "video", "video HD", "poster"][1]
    if MODE == "sketchy":
        dim_xy = (72, 54)
        supersampling = 1
        max_iter = 1024
        fps = 10
    elif MODE == "video LD":
        dim_xy = (720, 540)
        supersampling = 1
        max_iter = 8192
        fps = 10
    elif MODE == "video":
        dim_xy = (720, 540)
        supersampling = 3
        max_iter = 8192
        fps = 30
    elif MODE == "video HD":
        dim_xy = (1920, 1080)
        supersampling = 3
        max_iter = 8192
        fps = 60
    elif MODE == "poster":
        dim_xy = (6000, 4500)
        supersampling = 3
        max_iter = 8192
        fps = 0
    else:
        dim_xy = (1, 1)
        supersampling = 1
        max_iter = 1
        fps = 0

    itinary = sound_itinary_planner.Itinary(sound_itinary_planner.AnotherPlanetMap(), fps)
    generate_hits_from_itinary(data_folder, dim_xy, itinary.full_itinary, supersampling, max_iter)
    generate_images_from_hits(data_folder, itinary.full_itinary, max_iter, fps)
    generate_video_from_folder(data_folder, fps)

if __name__ == '__main__':
    main()
