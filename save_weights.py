import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.model_utils import load_models, load_model_from_torch
from utils.img_utils import img_loader, disp_loader, pad_img, numpy_split_like_torch

import tensorflow as tf

from models.refiner import Refiner

import math


def save():
    fe, classifier, regressor = load_model_from_torch(weight_path="utils/net_latest.pt")

    fe.save_weights("weights/fe/fe")
    classifier.save_weights("weights/classifier/classifier")
    regressor.save_weights("weights/regressor/regressor")

    fe, classifier, regressor = load_models(weight_dir="weights")

    # refiner = Refiner(config=None, backbone=fe, classifier=classifier, regressor=regressor, height=height, width=width)


if __name__ == '__main__':
    save()
