import fractal_painter
import numpy as np

def assert_same_arrays(a, b, eps = 0):
    square_difference = np.sum(np.square(a-b))
    try:
        assert(square_difference <= eps)
    except AssertionError:
        print("AssertionError: square_difference =", square_difference, " is greater than", eps, flush=True)
        raise AssertionError

def test_apply_color_map():
    # (M x N)
    hits = np.array([
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
    ])

    # (N x 1 x 3)
    colorbar = np.array([
        [[0, 0, 0]],
        [[1, 1, 1]],
        [[2, 2, 2]],
        [[3, 3, 3]],
    ])

    actual_res = fractal_painter.apply_color_map(hits, colorbar)
    expected_res = np.array([[[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]], [[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]]])
    assert_same_arrays(actual_res, expected_res)
    print("test_apply_color_map: SUCCESS")

def main():
    test_apply_color_map()

if __name__ == '__main__':
    main()