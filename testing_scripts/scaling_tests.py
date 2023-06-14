import numpy as np


def scale_coords_tf(points, min_x, max_x, a, b):
    # https: // en.wikipedia.org / wiki / Feature_scaling
    return a + (((points - min_x) * (b - a)) / (max_x - min_x))


def scale_coords_fixed(points, max_length):
    return -1 + (2 * points / (max_length - 0.0))


def scale_coords_original(points, max_length):
    return 2 * points / (max_length - 1.0) - 1.0


def main():
    test_1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    test_2 = np.array([2.5, 5.23, 6.115, 8.66, 7.25, 9.85])
    # test_2 = np.array[100,101,102,103,104,105,106,107,108,109,110]

    scaled_original = scale_coords_original(points=test_1, max_length=10)
    scaled_fixed = scale_coords_fixed(points=test_1, max_length=10)
    scaled_tf = scale_coords_tf(points=test_1, min_x=0, max_x=10, a=-1, b=1)

    print(f"Original: {scaled_original}")
    print(f"Fixed: {scaled_fixed}")
    print(f"TF: {scaled_tf}")
    print("\n\n")
    scaled_original = scale_coords_original(points=test_2, max_length=10)
    scaled_fixed = scale_coords_fixed(points=test_2, max_length=10)
    scaled_tf = scale_coords_tf(points=test_2, min_x=0, max_x=10, a=-1, b=1)

    print(f"Original: {scaled_original}")
    print(f"Fixed: {scaled_fixed}")
    print(f"TF: {scaled_tf}")

if __name__ == '__main__':
    main()
