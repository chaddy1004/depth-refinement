import torch
import tensorflow_graphics as tfg
from tensorflow_graphics.image import transformer
import numpy as np
import tensorflow as tf


# def scale_coords_tf(points, min_x, max_x, a, b):
#     # https: // en.wikipedia.org / wiki / Feature_scaling
#     return a + (((points - min_x) * (b - a)) / (max_x - min_x))


def scale_coords_original(points, max_length):
    return 2 * points / (max_length - 1.0) - 1.0

def scale_coords_tf(points, min_x, max_x, a, b):
    # https: // en.wikipedia.org / wiki / Feature_scaling
    return a + (((points - min_x) * (b - a)) / (max_x - min_x))


def scale_coords_tf_matching_original_paper(points, min_x, max_x, a, b):
    # https: // en.wikipedia.org / wiki / Feature_scaling
    return a + (((points - min_x) * (b - a)) / (max_x + a))


def scale_coords(points, max_length):
    return -1 + (2 * points / (max_length - 0.0))


def tensorflow_interpolate(features, uv):
    uv = np.transpose(a=uv, axes=(0, 2, 1))
    uv = uv[:, :, np.newaxis, :]
    # 0 is full integer, 1 is half integer
    samples = transformer.sample(image=features, warp=uv)
    return samples[:, :, 0, :]


def pytorch_interpolate(features, uv):
    # print(uv.shape)
    uv = np.transpose(a=uv, axes=(0, 2, 1))
    uv = uv[:, :, np.newaxis, :]
    uv = torch.from_numpy(uv)
    samples = torch.nn.functional.grid_sample(input=features.double(), grid=uv.double(), mode="bilinear",
                                              padding_mode="zeros",
                                              align_corners=None)
    return samples[:, :, :, 0]


def numpy_split_like_torch(array, size_of_chunk, array_length_dim):
    len_of_array = np.shape(array)[array_length_dim]
    return np.split(ary=array, indices_or_sections=range(size_of_chunk, len_of_array, size_of_chunk),
                    axis=array_length_dim)


def main(height=10, width=20, num_samples=12, batch_size=1):
    height = int(height)
    width = int(width)
    nx = np.linspace(0, width, width)  #
    ny = np.linspace(0, height, height)
    u, v = np.meshgrid(nx, ny)

    coords = np.concatenate([u.flatten()[..., np.newaxis], v.flatten()[..., np.newaxis]], axis=1)
    coords = coords[np.newaxis, ...]
    batch_size, n_pts, _ = coords.shape
    coords = torch.Tensor(coords).float()

    # coordinate_subsets = np.array_split(
    #     ary=coords.numpy(), indices_or_sections=int(num_samples / batch_size), axis=1
    # )

    coordinate_subsets = numpy_split_like_torch(array=coords.numpy(), size_of_chunk=int(num_samples / batch_size),
                                                array_length_dim=1)

    coordinate_subsets_torch = torch.split(
        coords, int(num_samples / batch_size), dim=1
    )

    for i in range(len(coordinate_subsets)):
        print("np split vs torch split", i,
              np.all(np.isclose(a=coordinate_subsets[i], b=coordinate_subsets_torch[i].numpy())))
    feature_height = 256
    feature_width = 368
    feature_ch = 64
    dummy_feature_ch_first = np.random.rand(1, feature_ch, feature_height, feature_width)
    dummy_feature_ch_last = np.transpose(dummy_feature_ch_first, axes=(0, 2, 3, 1))

    dummy_feature_pytorch = torch.from_numpy(dummy_feature_ch_first)
    dummy_feature_tensorflow = tf.convert_to_tensor(dummy_feature_ch_last)

    print("n_subsets", np.ceil(n_pts / num_samples))
    with torch.no_grad():
        for i, coordinate_subset in enumerate(coordinate_subsets):
            print("gridsample enumerate i", i)
            points = np.transpose(a=coordinate_subset, axes=(0, 2, 1))  # swap dim1 and dim2

            # u is width, x
            u_scaled = scale_coords(points=points[:, [0], :], max_length=width)  # (batch_size, 1, n_points)
            # v is width, y
            v_scaled = scale_coords(points=points[:, [1], :], max_length=height)  # (batch_size, 1, n_points)
            uv_scaled = np.concatenate([u_scaled, v_scaled], axis=1)

            # unlike torch grid sample, it must be scaled to the dim of the input
            u_scaled_tf = scale_coords_tf(points=points[:, [0], :], min_x=0, max_x=width, a=0,
                                          b=feature_width)  # (batch_size, 1, n_points)

            v_scaled_tf = scale_coords_tf(points=points[:, [1], :], min_x=0, max_x=height, a=0,
                                          b=feature_height)  # (batch_size, 1, n_points)

            uv_scaled_tf = np.concatenate([u_scaled_tf, v_scaled_tf], axis=1)
            # uv = np.concatenate([u, v], axis=1)
            pytorch_interpolated = pytorch_interpolate(features=dummy_feature_pytorch, uv=uv_scaled)

            # you need to put non-scaled coordinates for the tensorflow implemenation
            tensorflow_interpolated = tensorflow_interpolate(features=dummy_feature_tensorflow, uv=uv_scaled_tf)

            pytorch_interpolated = pytorch_interpolated.numpy()

            pytorch_interpolated = np.transpose(pytorch_interpolated, axes=(0, 2, 1))
            tensorflow_interpolated = tensorflow_interpolated.numpy()

            results = np.all(np.isclose(pytorch_interpolated, tensorflow_interpolated, rtol=1e-4))
            print(f"Comparison results {results}")


if __name__ == '__main__':
    main()
