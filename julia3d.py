import numpy as np
from numba import jit
import utils
import time

@jit
def q_square(qx, qy, qz, qw):
    x = 2*qx*qw
    y = 2*qy*qw
    z = 2*qz*qw
    w = -qx**2 - qy**2 - qz**2 + qw**2
    return [x, y, z, w]

@jit
def q_add(x0, y0, z0, w0, x1, y1, z1, w1):
    qout = list()
    qout.append(x0 + x1)
    qout.append(y0 + y1)
    qout.append(z0 + z1)
    qout.append(w0 + w1)
    return qout


def iterations(range0, qc, julia_hits, radius):
    qc = qc[0], qc[1], qc[2], qc[3]
    radius2 = radius**2
    for k, z in enumerate([0]):
        for j, y in enumerate(range0):
            for i, x in enumerate(range0):
                w2 = radius2 - (x**2+y**2+z**2)
                if w2 > 0:
                    q = [x, y, z, w2**0.5]
                    for n in range(127, 255):
                        w = q[3]
                        q = q_square(*q_square(*q_square(*q)))
                        if 2*w**2-q[3] > 8:
                            break
                        q = q_add(*q, *qc)
                    julia_hits[k, j, i] = n
                else:
                    julia_hits[k, j, i] = 63


def main():

    zoom = 5
    N = 100

    ratio = 2 / N / zoom
    julia_hits = np.zeros(shape=(1, N, N), dtype=np.uint8)

    range0 = np.linspace(-N / 2, N / 2, N) * ratio
    for r in range(10):
        tic = time.time()
        xyz = np.random.random(3)
        w = (1-sum(xyz**2))**0.5
        qc = np.array([xyz[0], xyz[1], xyz[2], w])
        print(qc)
        iterations(range0, qc, julia_hits, 1)

        print("elapsed:", time.time()-tic, "s")
        # for n in np.linspace(0, N-1, 10):
        #     utils.export_to_png(f"julia3d_{int(n)}", julia_hits[int(n), :, :])
        utils.export_to_png(f"julia3d_{r}", julia_hits[0, :, :])

if __name__ == '__main__':
    main()
