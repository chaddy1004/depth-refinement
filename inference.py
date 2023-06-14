import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.model_utils import load_models
from utils.img_utils import img_loader, disp_loader, pad_img, numpy_split_like_torch

import tensorflow as tf

from models.refiner import Refiner

import math


def inference(refiner, height, width, pad, x, num_samples=50000):
    height = int(height)
    width = int(width)
    start_x = pad[2]
    end_x = (x[0].shape[1] - pad[3])
    start_y = pad[0]
    end_y = (x[0].shape[0] - pad[1])
    nx = np.linspace(start_x, end_x, width)  #
    ny = np.linspace(start_y, end_y, height)
    u, v = np.meshgrid(nx, ny)

    coords = np.concatenate([u.flatten()[..., np.newaxis], v.flatten()[..., np.newaxis]], axis=1)
    coords = coords[np.newaxis, ...]
    batch_size, n_pts, _ = coords.shape

    coordinate_subsets = numpy_split_like_torch(array=coords, size_of_chunk=n_pts,
                                                array_length_dim=1)

    out_preds = []
    out_confidences = []
    for i, coordinate_subset in enumerate(coordinate_subsets):
        print("enumerate i", i)
        points = np.transpose(a=coordinate_subset, axes=(0, 2, 1))  # swap dim1 and dim2
        refiner.query(points=points)
        preds = refiner.get_disparity()
        confidence = refiner.get_confidence()
        preds = tf.squeeze(preds)
        confidence = tf.squeeze(confidence)
        # if preds.shape[0] < num_samples:
        #     preds = tf.concat([preds, tf.zeros((num_samples - preds.shape[0]))], axis=0)
        # if confidence.shape[0] < num_samples:
        #     confidence = tf.concat([confidence, tf.zeros((num_samples - confidence.shape[0]))], axis=0)

        out_preds.append(preds)
        out_confidences.append(confidence)

    out_preds = tf.stack(out_preds)
    out_confidences = tf.stack(out_confidences)
    output = tf.stack([out_preds, out_confidences])

    res = []
    for i in range(2):
        flattened = tf.reshape(output[i, ...], (1, -1))
        flattened = flattened[0, :n_pts]
        image_shaped = tf.reshape(flattened, (height, width, -1))
        res.append(image_shaped)

    return res


def test(image_path, disp_path, max_disp=256):
    rgb = img_loader(image_path)
    height, width = rgb.shape[:2]

    fe, classifier, regressor = load_models(weight_dir="weights")

    refiner = Refiner(config=None, backbone=fe, classifier=classifier, regressor=regressor, height=height, width=width)

    disp = disp_loader(disp_path, 256) / 1.0
    disp[disp > max_disp] = 0
    height_disp, width_disp = disp.shape[:2]
    rgb, pad = pad_img(rgb, height=height, width=width, divisor=32)
    disp, _ = pad_img(disp, height=height, width=width, divisor=32)

    rgb = rgb.astype(float)
    disp = disp.astype(float)
    rgb_tensor = tf.convert_to_tensor(rgb)
    disp_tensor = tf.convert_to_tensor(disp)
    o_shape_tensor = tf.convert_to_tensor(np.asarray((height, width)))

    if len(rgb_tensor.shape) == 3:
        rgb_tensor = tf.expand_dims(rgb_tensor, 0)
    if len(disp_tensor.shape) == 3:
        disp_tensor = tf.expand_dims(disp_tensor, 0)

    x = tf.concat([rgb_tensor, disp_tensor], axis=-1)
    refiner.feature_extraction(x)
    res = inference(refiner=refiner, height=height, width=width, pad=pad, x=x)
    pred = res[0].numpy()
    confidence = res[1].numpy()

    print(pred, pred.shape, np.max(pred), np.min(pred), (pred.shape[1] / width_disp))
    pred = pred * (pred.shape[1] / width_disp)
    pred = pred

    os.makedirs("output", exist_ok=True)
    pred = np.squeeze(pred)
    confidence = np.squeeze(confidence)
    pred = cv2.resize(pred, (width, height))
    img_name = image_path.split("/")[-1]
    img_name = img_name.replace("rgb", "refined")
    plt.imsave(
        f"samples/outputs/{img_name}",
        pred,
        cmap="magma"
    )

    # plt.imsave(
    #     "refined_confidence.png",
    #     confidence,
    #     cmap="magma",
    # )


if __name__ == '__main__':
    for dataset in ["cones", "chair", "teddy"]:
        for algorithm in ["AC", "BM", "SGM"]:
            test(image_path=f"samples/outputs/{dataset}_{algorithm}_rgb.png",
                 disp_path=f"samples/outputs/{dataset}_{algorithm}_disp.png")
