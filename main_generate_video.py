import pickle
import juliaset
import fractal_painter
import utils
import os
import cv2 as cv

def generate_video_from_folder():
    folder = "gallery/julia_stop_motion/"
    img0 = cv.imread(folder + "0.png")
    H, W, _ = img0.shape

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv.VideoWriter("gallery/juliaset.mp4", fourcc, 20.0, (W, H))

    for k in range(len(os.listdir(folder))):
        image_path = folder + str(k) + ".png"
        frame = cv.imread(image_path)

        out.write(frame)  # Write out frame to video

        print(k)
    # Release everything if job is finished
    out.release()

def generate_images_from_hits():
    print("go")
    with open("hits.pkl", "rb") as pickle_in:
        all_hits = pickle.load(pickle_in)
    for k, julia_hits in enumerate(all_hits):
        julia_bgr = fractal_painter.color_map(julia_hits)
        julia_bgr = fractal_painter.glow_effect(julia_bgr)
        utils.export_to_png(f"julia_stop_motion/{k}", julia_bgr)

def interpolate_locations(locA, locB, t):
    out = {}
    for keyword in ["pos_julia_xy", "pos_mandel_xy"]:
        x = locA[keyword][0] * (1 - t) + t * locB[keyword][0]
        y = locA[keyword][1] * (1 - t) + t * locB[keyword][1]
        out[keyword] = x, y

    for keyword in ["zoom", "r_mat"]:
        out[keyword] = locA[keyword] * (1 - t) + t * locB[keyword]

    return out

def generate_hits_from_itinary():
    nb_inter_frame = 3
    dim_xy = (720, 540)

    with open("itinary.pkl", "rb") as pickle_in:
        itinary = pickle.load(pickle_in)

    print(itinary)

    all_hits = []
    for i in range(len(itinary)-1):
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
                supersampling=1)
            all_hits.append(julia_hits)
    with open("hits.pkl", "wb") as pickle_out:
        pickle.dump(all_hits, pickle_out)
    print("ok")



if __name__ == '__main__':
    generate_hits_from_itinary()
    generate_images_from_hits()
    generate_video_from_folder()
