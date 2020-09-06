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


def load_sound_command(fps):
    sr, cmd = scipy.io.wavfile.read('sounds/Synthwave_200627.wav')
    cmd = np.max(np.abs(cmd), axis=1)

    block_size = int(round(sr/fps))
    length = len(cmd)
    block_nb = (length//block_size)
    cmd = cmd[:block_nb*block_size]
    cmd = np.reshape(cmd, newshape=(block_nb, block_size))
    cmd = np.max(cmd, axis=1)
    cmd = cmd.astype(np.float)/(2**15-1)
    return cmd


def generate_video_from_folder(data_folder, fps):
    if fps <= 0:
        print("fps must be a positive number")
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


def interpolate_locations(locA, locB, t):
    out = {}
    for keyword in ["pos_julia_xy", "pos_mandel_xy"]:
        x = locA[keyword][0] * (1 - t) + t * locB[keyword][0]
        y = locA[keyword][1] * (1 - t) + t * locB[keyword][1]
        out[keyword] = x, y

    for keyword in ["zoom", "fisheye_factor"]:  #, "r_mat",]:
        out[keyword] = locA[keyword] * (1 - t) + t * locB[keyword]
    out["r_mat"] = np.eye(4)
    return out


def estimate_computation_time(itinary, dim_xy, nb_inter_frame, supersampling):
    time_per_pixels = [location["time_per_px"] for location in itinary]
    time_per_pixels[0] *= 0.5
    time_per_pixels[-1] *= 0.5
    avg_time_per_pixels = sum(time_per_pixels)/(len(time_per_pixels)-1)
    W, H = dim_xy
    out = avg_time_per_pixels * W*H * nb_inter_frame * (len(itinary)-1) * supersampling**2
    return out


def generate_full_itinary(data_folder, cmd, fps):
    print("generate_full_itinary")
    tic0 = time.time()

    with open(pth(data_folder, "itinary.pkl"), "rb") as pickle_in:
        itinary = pickle.load(pickle_in)

    nb_block = len(cmd)
    # index of itinaries, except first and last ones
    itinary_at_samples = [2073661, 3915919, 4792857, 5000000, 6474654]
    itinary_at_frame = [sample//48000 for sample in itinary_at_samples]  # TODO: recover 48000 from somewhere
    itinary_at_frame = [0] + itinary_at_frame + [nb_block-1]

    if len(itinary) != len(itinary_at_frame):
        print("Error")
        raise Exception  # TODO: raise something

    k = 0
    locations = []
    prev_loc = itinary[0]
    for i in range(len(itinary) - 1):
        locA = itinary[i]
        locB = itinary[i + 1]
        frameA = itinary_at_frame[i]
        frameB = itinary_at_frame[i + 1]
        nb_inter_frame = frameB-frameA
        for j in range(nb_inter_frame):
            t = j / nb_inter_frame
            loc = interpolate_locations(locA, locB, t)
            loc["cmd"] = cmd[k]

            xyz0 = np.append(loc["pos_julia_xy"], loc["zoom"])
            xyz1 = np.append(prev_loc["pos_julia_xy"], prev_loc["zoom"])
            velocity = np.sum((xyz1-xyz0)**2)**0.5*fps
            loc["fisheye"] = -velocity

            locations.append(loc)
            prev_loc = loc.copy()
            k += 1
    return locations



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


if __name__ == '__main__':
    data_folder = "output2"

    MODE = ["sketchy", "video LD", "video", "video HD", "poster"][1]
    if MODE == "sketchy":
        dim_xy = (72, 54)
        nb_inter_frame = 10
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

    cmd = load_sound_command(fps)
    full_itinary = generate_full_itinary(data_folder, cmd, fps)
    generate_hits_from_itinary(data_folder, dim_xy, full_itinary, supersampling, max_iter)
    generate_images_from_hits(data_folder, full_itinary, max_iter)
    generate_video_from_folder(data_folder, fps)
