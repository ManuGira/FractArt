import pickle
import juliaset

def generate_images_from_hits():
    print("go")
    with open("hits.pkl", "rb") as pickle_in:
        all_hits = pickle.load(pickle_in)
    for julia_hits in all_hits:
        print("ok")
        # TODO:

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
    dim_xy = (1000, 1000)

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
                location["pos_mandel_xy"])
            all_hits.append(julia_hits)
    with open("hits.pkl", "wb") as pickle_out:
        pickle.dump(all_hits, pickle_out)
    print("ok")



if __name__ == '__main__':
    # generate_hits_from_itinary()
    generate_images_from_hits()
