import time
import numpy as np
import juliaset

def juliaset_perf(location):
    """
    dim_xy = (100, 100)
    supersampling = 3
    max_iter = 8196
        juliaset_vectorized     1227 Mi/s -> score 9.1
        juliaset_njit            218 Mi/s -> score 8.3
        juliaset_numpy           0.8 Mi/s -> score 5.9
    """
    # to measure image computation time
    dim_xy = (100, 100)
    supersampling = 3
    max_iter = 8196

    julia_func = juliaset.juliaset_vectorized
    # julia_func = juliaset.juliaset_njit
    # julia_func = juliaset.juliaset_numpy

    # to measure transfer time
    args_transfer = [
        dim_xy,
        location["pos_julia_xy"],
        location["zoom"],
        location["r_mat"],
        location["pos_mandel_xy"],
        supersampling,
        location["fisheye_factor"],
        1,
    ]

    args_image = [
        dim_xy,
        location["pos_julia_xy"],
        location["zoom"],
        location["r_mat"],
        location["pos_mandel_xy"],
        supersampling,
        location["fisheye_factor"],
        max_iter,
    ]
    N = dim_xy[0]*dim_xy[1]

    # measure time which is not part of the algo itself, it is common to all algo
    print("Measure TRANSFORM time...")
    juliaset.juliaset_none(*args_image)
    tic0 = time.time()
    juliaset.juliaset_none(*args_image)
    tic1 = time.time()
    transform_dt = tic1-tic0
    print(f"{N} pixels in {transform_dt} seconds")
    print(f"{N/transform_dt/1e6:.2f} Mp/s\n")

    print("measure TRANSFER time...")
    julia_func(*args_transfer)
    tic0 = time.time()
    julia_func(*args_transfer)
    tic1 = time.time()
    transfer_dt = tic1 - tic0 - transform_dt
    print(f"{N} pixels in {transfer_dt} seconds")
    print(f"{N/transfer_dt/1e6:.2f} Mp/s\n")

    print("generate_hits_from_itinary...")
    julia_hits = julia_func(*args_image)
    tic0 = time.time()
    julia_hits = julia_func(*args_image)
    tic1 = time.time()
    dt = tic1 - tic0 - transform_dt - transfer_dt

    N = supersampling**2 * np.sum(julia_hits)
    print(N, "iteration in", dt, "seconds")
    print(int(N/dt/1e6), "Mi/s")
    score = np.log10(N/dt)
    print(f"score: {score:.1f}")
    return score

def main():
    location = {'pos_julia_xy':(-0.07015860596076647,-0.050134045790274674),'zoom':6,'r_mat':np.array([[0.99985864,0.,0.0168139,0.],[0.,1.,0.,0.],[-0.0168139,0.,0.99985864,0.],[0.,0.,0.,1.]]),'pos_mandel_xy':(-0.8408004442917478,-0.19367697713854448),'fisheye_factor':-0.9999999999999991,'time_per_px':3.9491994177246095e-07}
    juliaset_perf(location)


if __name__ == '__main__':
    main()