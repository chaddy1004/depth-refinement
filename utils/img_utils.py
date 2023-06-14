import math
import os
import re
import sys

import cv2
import numpy as np
import tensorflow as tf
import torch

from tensorflow_graphics.image import transformer


def numpy_split_like_torch(array, size_of_chunk, array_length_dim):
    len_of_array = np.shape(array)[array_length_dim]
    return np.split(ary=array, indices_or_sections=range(size_of_chunk, len_of_array, size_of_chunk),
                    axis=array_length_dim)

def tensorflow_interpolate(features, uv):
    uv = np.transpose(a=uv, axes=(0, 2, 1))
    uv = uv[:, :, np.newaxis, :]
    # 0 is full integer, 1 is half integer
    samples = transformer.sample(image=features, warp=uv, pixel_type=transformer.PixelType.HALF_INTEGER)
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


def readPFM(file):
    file = open(file, "rb")
    header = file.readline().rstrip()

    if (sys.version[0]) == "3":
        header = header.decode("utf-8")
    if header == "PF":
        color = True
    elif header == "Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    if (sys.version[0]) == "3":
        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("utf-8"))
    else:
        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    if (sys.version[0]) == "3":
        scale = float(file.readline().rstrip().decode("utf-8"))
    else:
        scale = float(file.readline().rstrip())

    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

def img_loader(path, mode="passive", height=2160, width=3840):
    img = None
    if not os.path.exists(path):
        raise ValueError(f"Cannot open image: {path}")
    if path.endswith("raw"):
        img = (
            np.fromfile(open(path, "rb"), dtype=np.uint8).reshape(height, width, 3)
            if mode == "passive"
            else np.fromfile(open(path, "rb"), dtype=np.uint8).reshape(height, width, 1)
        )
    else:
        img = cv2.imread(path, 1)
        if img.ndim == 2:
            img = np.expand_dims(img, -1)  #

    return img

def disp_loader(path, scale_factor16bit=256):
    disp = None
    if not os.path.exists(path):
        raise ValueError("Cannot open disp: " + path)

    if path.endswith("pfm"):
        disp = np.expand_dims(readPFM(path), 0)
    if path.endswith("png"):
        disp = np.expand_dims(cv2.imread(path, -1), 0)
        if disp.dtype == np.uint16:
            disp = disp / float(scale_factor16bit)
    if path.endswith("npy"):
        disp = np.expand_dims(np.load(path, mmap_mode="c"), 0)
    if disp is None:
        raise ValueError("Problems while loading the disp")
    # Remove invalid values
    disp[np.isinf(disp)] = 0

    return disp.transpose(1, 2, 0).astype(np.float32)

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

def get_coords(height: int, width: int, batch_size: int = 1):
    # split width into width number of samples (should the end really be width? woudlnt it not be counted?)
    nx = np.linspace(start=0, stop=width, num=width)
    ny = np.linspace(start=0, stop=height, num=height)
    u, v = np.meshgrid(nx, ny)
    coords = np.expand_dims(np.stack((u.flatten(), v.flatten()), axis=-1), 0)
    coords_batched = np.concatenate([coords for _ in range(batch_size)], axis=0)
    return coords_batched


def prepare_query_input(output_disp_height: int, output_disp_width: int, batch_size):
    coords = get_coords(height=output_disp_height, width=output_disp_width, batch_size=batch_size)
    batch_size, n_pts, _ = coords.shape
    coords_tensor = tf.convert_to_tensor(coords)
    return coords_tensor


def prepare_query_input_torch(output_disp_height: int, output_disp_width: int, batch_size, num_samples=200000,
                              num_out=2):
    coords = get_coords(height=output_disp_height, width=output_disp_width, batch_size=batch_size)
    # n_points is width *  height
    batch_size, n_points, _ = coords.shape

    coords = torch.Tensor(coords).float().to(device="cpu") # shape = (batch, n_points, 2)
    coords_changed = torch.reshape(coords, (batch_size, -1, 2))

    # 0th output is are the
    output = torch.zeros(num_out, math.ceil(output_disp_width * output_disp_height / num_samples), num_samples)

    split = torch.split(
        coords.reshape(batch_size, -1, 2), int(num_samples / batch_size), dim=1
    )
    with torch.no_grad():
        for i, p_split in enumerate(split):
            points = torch.transpose(p_split, 1, 2)
            print(points)
            # net.query(points.to(device=cuda))
            # preds = net.get_disparity()
            # confidence = net.get_confidence()
            # output1 = output[0, i, : p_split.shape[1]] = preds.to(device=cuda)
            # output2 = output[1, i, : p_split.shape[1]] = confidence.to(device=cuda)
    # res = []
    # for i in range(num_out):
    #     res.append(output[i].view(1, -1)[:, :n_pts].reshape(-1, height, width))
    # return res
    return output


def pad_img(img: np.ndarray, height: int = 1024, width: int = 1024, divisor: int = 32):
    """Pad the input image, making it larger at least (:attr:`height`, :attr:`width`)

    Params:
    ----------

    img (np.ndarray):
        array with shape h x w x c

    height (int):
        new minimum height

    width (int):
        new minimum width

    divisor (int):
        divisor factor, it forces the padded array to be multiple of divisor

    Returns:
        a new array with shape  H x W x c, multiple of divisior, and
        the amount of padding
    """
    h_pad = 0 if (height % divisor) == 0 else divisor - (height % divisor)
    top = h_pad // 2
    bottom = h_pad - top
    w_pad = 0 if (width % divisor) == 0 else divisor - (width % divisor)
    left = w_pad // 2
    right = w_pad - left
    img = np.lib.pad(img, ((top, bottom), (left, right), (0, 0)), mode="reflect")
    pad = np.stack([top, bottom, left, right], axis=0)
    return img, pad

def depad_img(
        img: np.ndarray,
        pad: np.ndarray,
        upsampling_factor: float = 1,
):
    """Remove padding from tensor

    Params:
    -------------

    img (np.ndarray):
        array to de-pad, with shape CxHxW or HxW
    pad (np.ndarray):
        array (top_pad, bottom_pad, left_pad, right_pad) with shape 1x4
    upsampling_factor (int):
        how to scale crops. For instance, if :attr:`upsampling_factor: is 4,
        crops are upscaled by 4.
        Default is 1.
    Returns:
    ------------
        a np.ndarray
    """

    if not img.ndim == 3:
        img = np.expand_dims(img, 0)
    pad = pad.squeeze()
    top = int(pad[0] * upsampling_factor)
    bottom = int(pad[1] * upsampling_factor)
    left = int(pad[2] * upsampling_factor)
    right = int(pad[3] * upsampling_factor)

    return img[
           :,
           top: img.shape[1] - bottom,
           left: img.shape[2] - right,
           ]


if __name__ == '__main__':
    # prepare_query_input(output_disp_height=10, output_disp_width=20, batch_size=8)
    prepare_query_input_torch(output_disp_height=10, output_disp_width=20, batch_size=8)
