import tensorflow as tf
import torch
from tensorflow import keras
from keras.layers import Layer, Conv1D, Softmax
from keras import Model

from utils.activation_utils import Sine
from utils.model_utils import get_backbone

from models.classifier import Classifier
from models.regressor import Regressor

from utils.img_utils import scale_coords_tf_matching_original_paper, tensorflow_interpolate ,scale_coords_tf, scale_coords_original, pytorch_interpolate
import numpy as np


class Refiner(Model):
    def __init__(self, config, backbone, classifier, regressor, height, width):
        super(Refiner, self).__init__()
        self.config = config
        self.backbone = backbone
        self.height = None
        self.width = None

        self.classifier = classifier
        self.regressor = regressor

    def feature_extraction(self, batch):
        self.feat_list = self.backbone(batch)
        _, height, width, _ = batch.shape
        self.height = height
        self.width = width

    def query(self, points):
        features = None
        for i, im_feat in enumerate(self.feat_list):
            if i == 0:
                u_scaled_tf = scale_coords_tf_matching_original_paper(points=points[:, [0], :], min_x=0,
                                                                      max_x=self.width, a=0,
                                                                      b=im_feat.shape[
                                                                            2] + 0.5)  # (batch_size, 1, n_points)

                v_scaled_tf = scale_coords_tf_matching_original_paper(points=points[:, [1], :], min_x=0,
                                                                      max_x=self.height, a=0,
                                                                      b=im_feat.shape[
                                                                            1] + 0.5)  # (batch_size, 1, n_points)
            else:
                u_scaled_tf = scale_coords_tf_matching_original_paper(points=points[:, [0], :], min_x=0,
                                                                      max_x=self.width, a=0,
                                                                      b=im_feat.shape[2]+1)  # (batch_size, 1, n_points)

                v_scaled_tf = scale_coords_tf_matching_original_paper(points=points[:, [1], :], min_x=0,
                                                                      max_x=self.height, a=0,
                                                                      b=im_feat.shape[1]+1)  # (batch_size, 1, n_points)

            # u_scaled_tf = scale_coords_tf_matching_original_paper(points=points[:, [0], :], min_x=0,
            #                                                       max_x=self.width, a=0,
            #                                                       b=im_feat.shape[2])  # (batch_size, 1, n_points)
            #
            # v_scaled_tf = scale_coords_tf_matching_original_paper(points=points[:, [1], :], min_x=0,
            #                                                       max_x=self.height, a=0,
            #                                                       b=im_feat.shape[1])  # (batch_size, 1, n_points)
            uv_scaled_tf = np.concatenate([u_scaled_tf, v_scaled_tf], axis=1)

            # u_scaled = scale_coords_original(points=points[:, [0], :], max_length=im_feat.shape[2])  # (batch_size, 1, n_points)
            # v is width, y
            # v_scaled = scale_coords_original(points=points[:, [1], :], max_length=im_feat.shape[1])  # (batch_size, 1, n_points)
            # uv_scaled = np.concatenate([u_scaled, v_scaled], axis=1)

            # pytorch_interpolated = pytorch_interpolate(features=torch.Tensor(im_feat.numpy().transpose(0, 3, 1, 2)), uv=uv_scaled)

            # you need to put non-scaled coordinates for the tensorflow implemenation
            # tensorflow_interpolated = tensorflow_interpolate(features=im_feat, uv=uv_scaled_tf)

            # pytorch_interpolated = pytorch_interpolated.numpy()
            #
            # pytorch_interpolated = np.transpose(pytorch_interpolated, axes=(0, 2, 1))
            # tensorflow_interpolated = tensorflow_interpolated.numpy()

            # interp_feat is different!
            # im_feat is the same
            # why is this happeningggggg

            # may 31st morning -> This is the same now
            interp_feat = tensorflow_interpolate(features=im_feat, uv=uv_scaled_tf)
            if i == 0:
                features = interp_feat
            else:
                features = tf.concat([features, interp_feat], -1)

        # this is where it is off now as of may 31st morning
        self.probs, feat1, feat2, feat3, feat4 = self.classifier(features)
        self.disparity = (
            tf.argmax(self.probs, axis=-1).numpy()
        )
        self.disparity = tf.expand_dims(self.disparity, axis=-1)

        # self.probs = tf.cast(self.probs, self.probs.dtype)
        self.disparity = tf.cast(self.disparity, self.probs.dtype)

        offset = self.regressor(tf.concat([features, self.disparity], -1))
        self.disparity = self.disparity + offset

    def get_disparity(self):
        return self.disparity

    def get_probs(self):
        return self.probs

    def get_confidence(self):
        confidence = tf.reduce_max(self.probs, axis=-1, keepdims=True)
        return confidence

    def call(self, inputs):
        pass
