import tensorflow as tf
from keras.applications import VGG16, ResNet50
from keras.layers import Layer
from keras.layers import Layer, Conv1D, Conv2D, Softmax, Activation, ReLU, MaxPool2D, Dense, Flatten, UpSampling2D
from keras import Sequential
from keras import Model
import torchvision
import torch
import numpy as np
from models.classifier import Classifier
from models.regressor import Regressor

from utils.img_utils import pad_img
import os
from glob import glob

BACKBONES = {"VGG16": VGG16(include_top=False), "ResNet50": ResNet50(include_top=False)}


def load_torch_statedict(backbone, mlp_classification, mlp_regression, statedict_path="net_latest.pt"):
    state_dict = torch.load(statedict_path, map_location="cpu")["state_dict"]
    for weight_name, weight_val in state_dict.items():
        position = None
        try:
            split = weight_name.split(".")
            if len(split) == 4:
                network_name, layer_name, position, weight_type = split
            elif len(split) == 3:
                network_name, layer_name, weight_type = split
            else:
                raise ValueError("name is not correct")
        except ValueError:
            print("Error", weight_name, position)
            return
        print(weight_name, position)
        if "downsample_" in weight_name:
            # to offset the positional value of VGG layer
            # the original code takes from the 4th layer of vgg which is why the name is also offsetted
            downsample_position = int(weight_name.split("_")[1])
            layer_position = np.log2(downsample_position)
            position = int(position) - (4 * layer_position + layer_position - 1)
        try:
            if position is not None:
                layer = eval(f"{network_name}.{layer_name}.layers[{int(position)}]")
            else:
                layer = eval(f"{network_name}.{layer_name}")
        except IndexError:
            print("ERROR VALUES: ", network_name, layer_name, position)
            return
        if weight_type == "weight":
            kernel = weight_val.detach().cpu().numpy()
            if len(kernel.shape) == 4:
                kernel = np.transpose(kernel, axes=(2, 3, 1, 0))
            elif len(kernel.shape) == 3:
                kernel = np.transpose(kernel, axes=(2, 1, 0))
            layer_weight = layer.weights[0]
            layer_weight.assign(kernel)
            assert "kernel" in layer_weight.name
        else:
            bias = weight_val.detach().cpu().numpy()
            layer_weight = layer.weights[1]
            layer_weight.assign(bias)
            assert "bias" in layer_weight.name
    print("LOADED TORCH MODEL")
    return backbone, mlp_classification, mlp_regression


def vgg13(num_classes):
    # this is based on the torchvision one that was used in the original paper
    model = tf.keras.Sequential()

    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeUniform(),
                     bias_initializer=tf.keras.initializers.Zeros()))
    model.add(ReLU())
    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeUniform(),
                     bias_initializer=tf.keras.initializers.Zeros()))
    model.add(ReLU())
    model.add(MaxPool2D((2, 2), strides=(2, 2)))

    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeUniform(),
                     bias_initializer=tf.keras.initializers.Zeros()))
    model.add(ReLU())
    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeUniform(),
                     bias_initializer=tf.keras.initializers.Zeros()))
    model.add(ReLU())
    model.add(MaxPool2D((2, 2), strides=(2, 2)))

    # Block 3
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeUniform(),
                     bias_initializer=tf.keras.initializers.Zeros()))
    model.add(ReLU())
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeUniform(),
                     bias_initializer=tf.keras.initializers.Zeros()))
    model.add(ReLU())
    model.add(MaxPool2D((2, 2), strides=(2, 2)))

    # Block 4
    model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeUniform(),
                     bias_initializer=tf.keras.initializers.Zeros()))
    model.add(ReLU())
    model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeUniform(),
                     bias_initializer=tf.keras.initializers.Zeros()))
    model.add(ReLU())
    model.add(MaxPool2D((2, 2), strides=(2, 2)))

    # Block 5
    model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeUniform(),
                     bias_initializer=tf.keras.initializers.Zeros()))
    model.add(ReLU())
    model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeUniform(),
                     bias_initializer=tf.keras.initializers.Zeros()))
    model.add(ReLU())
    model.add(MaxPool2D((2, 2), strides=(2, 2)))

    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(4096))
    model.add(ReLU())
    model.add(Dense(4096))
    model.add(ReLU())
    model.add(Dense(num_classes, activation='softmax'))

    return model


def get_backbone(config):
    try:
        return BACKBONES[config.exp.backbone]
    except KeyError:
        raise KeyError("Proper backbone needed")


class FeatureExtractor(Model):
    def __init__(self, config=None):
        super().__init__()
        # self.backbone = get_backbone(config)
        self.backbone = BACKBONES["VGG16"]
        self.max_disp = 256

        self.stem_block_depth = Sequential(
            [
                Conv2D(filters=64, kernel_size=3, padding="same", kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU(),
                MaxPool2D(),
                Conv2D(filters=64, kernel_size=3, padding="same", kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU()
            ]
        )

        # v99 4 to 24
        self.downsample_2_d = Sequential(
            [
                MaxPool2D((2, 2), strides=(2, 2)),
                Conv2D(128, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU(),
                Conv2D(128, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU()
            ]
        )

        self.downsample_4_d = Sequential(
            [
                MaxPool2D((2, 2), strides=(2, 2)),
                Conv2D(256, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU(),
                Conv2D(256, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU(),
            ]
        )

        self.downsample_8_d = Sequential(
            [
                MaxPool2D((2, 2), strides=(2, 2)),
                Conv2D(512, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU(),
                Conv2D(512, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU()
            ]
        )

        self.downsample_16_d = Sequential(
            [
                MaxPool2D((2, 2), strides=(2, 2)),
                Conv2D(512, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU(),
                Conv2D(512, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU(),
            ]
        )

        self.stem_block_rgb = Sequential(
            [
                Conv2D(filters=64, kernel_size=3, padding="same", kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU(),
                MaxPool2D(),
                Conv2D(filters=64, kernel_size=3, padding="same", kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU()
            ]
        )

        # v99 4 to 24
        self.downsample_2_rgb = Sequential(
            [
                MaxPool2D((2, 2), strides=(2, 2)),
                Conv2D(128, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU(),
                Conv2D(128, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU()
            ]
        )

        self.downsample_4_rgb = Sequential(
            [
                MaxPool2D((2, 2), strides=(2, 2)),
                Conv2D(256, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU(),
                Conv2D(256, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU(),
            ]
        )

        self.downsample_8_rgb = Sequential(
            [
                MaxPool2D((2, 2), strides=(2, 2)),
                Conv2D(512, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU(),
                Conv2D(512, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU()
            ]
        )

        self.downsample_16_rgb = Sequential(
            [
                MaxPool2D((2, 2), strides=(2, 2)),
                Conv2D(512, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU(),
                Conv2D(512, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU(),
            ]
        )

        self.upsample_16 = Sequential(
            [
                Conv2D(filters=512, kernel_size=3, padding="same", kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU(),
                Conv2D(filters=512, kernel_size=3, padding="same", kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU(),
                Conv2D(filters=512, kernel_size=3, padding="same", kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU(),
                UpSampling2D(size=(2, 2), interpolation="nearest"),
            ]
        )

        self.upsample_8 = Sequential(
            [
                Conv2D(filters=512, kernel_size=3, padding="same", kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU(),
                Conv2D(filters=256, kernel_size=3, padding="same", kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU(),
                Conv2D(filters=256, kernel_size=3, padding="same", kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU(),
                UpSampling2D(size=(2, 2), interpolation="nearest"),
            ]
        )

        self.upsample_4 = Sequential(
            [
                Conv2D(filters=256, kernel_size=3, padding="same", kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU(),
                Conv2D(filters=256, kernel_size=3, padding="same", kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU(),
                Conv2D(filters=128, kernel_size=3, padding="same", kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU(),
                UpSampling2D(size=(2, 2), interpolation="nearest"),
            ]
        )

        self.upsample_2 = Sequential(
            [
                Conv2D(filters=128, kernel_size=3, padding="same", kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU(),
                Conv2D(filters=128, kernel_size=3, padding="same", kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU(),
                Conv2D(filters=64, kernel_size=3, padding="same", kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU(),
                UpSampling2D(size=(2, 2), interpolation="nearest"),
            ]
        )

        self.final_conv = Sequential(
            [
                Conv2D(filters=64, kernel_size=3, padding="same", kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
                ReLU(),
                UpSampling2D(size=(2, 2), interpolation="nearest"),
                Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer=tf.keras.initializers.HeUniform(),
                       bias_initializer=tf.keras.initializers.Zeros()),
            ]
        )
        self.out_ch = 32 + 64

    def call(self, inputs, training=None):
        depth = inputs[:, :, :, -1]

        depth = tf.expand_dims(depth, axis=-1)
        depth = depth / self.max_disp
        stem_block_d = self.stem_block_depth(depth)
        downsample_2_d = self.downsample_2_d(stem_block_d)
        downsample_4_d = self.downsample_4_d(downsample_2_d)
        downsample_8_d = self.downsample_8_d(downsample_4_d)
        downsample_16_d = self.downsample_16_d(downsample_8_d)

        rgb = inputs[:, :, :, :-1]
        rgb = rgb / 255.0
        stem_block_rgb = self.stem_block_rgb(rgb)
        downsample_2_rgb = self.downsample_2_rgb(stem_block_rgb)
        downsample_4_rgb = self.downsample_4_rgb(downsample_2_rgb)
        downsample_8_rgb = self.downsample_8_rgb(downsample_4_rgb)
        downsample_16_rgb = self.downsample_16_rgb(downsample_8_rgb)

        upsample_16 = self.upsample_16(downsample_16_rgb + downsample_16_d)
        upsample_8 = self.upsample_8(upsample_16 + downsample_8_rgb + downsample_8_d)
        upsample_4 = self.upsample_4(upsample_8 + downsample_4_rgb + downsample_4_d)
        upsample_2 = self.upsample_2(upsample_4 + downsample_2_rgb + downsample_2_d)

        upsample_1 = self.final_conv(upsample_2)
        return [upsample_2, upsample_1]


def load_models():
    fe = FeatureExtractor()
    classifier = Classifier(filter_channels=[fe.out_ch, 512, 256, 128, fe.max_disp])
    regressor = Regressor(filter_channels=[fe.out_ch + 1, 128, 64, 1])

    classifier.build(input_shape=(None, 256, 368, 64 + 32))
    regressor.build(input_shape=(None, 256, 368, 64 + 32 + 1))
    fe.build(input_shape=(None, 512, 736, 4))


    weight_paths = ["weights/classifier", "weights/regressor", "weights/fe"]
    for weight_path in weight_paths:
        if "fe" in weight_path:
            fe.load_weights(weight_path)
        elif "classifier" in weight_path:
            classifier.load_weights(weight_path)
        elif "regressor" in weight_path:
            regressor.load_weights(weight_path)
        else:
            raise ValueError("WEight path name is not either fe, classifier, or regressor")

    return fe, classifier, regressor


def load_model_from_torch(weight_path=None):
    fe = FeatureExtractor()
    classifier = Classifier(filter_channels=[fe.out_ch, 512, 256, 128, fe.max_disp])
    regressor = Regressor(filter_channels=[fe.out_ch + 1, 128, 64, 1])

    if weight_path is not None:
        dummy_feature_ch_first = np.random.rand(64 + 32, 256, 368)
        dummy_feature_ch_last = np.transpose(dummy_feature_ch_first, axes=(1, 2, 0))
        dummy_input = tf.convert_to_tensor(dummy_feature_ch_last)
        dummy_input = tf.expand_dims(dummy_input, axis=0)
        _ = classifier(dummy_input, training=True)

        dummy_feature_ch_first = np.random.rand(64 + 32 + 1, 256, 368)
        dummy_feature_ch_last = np.transpose(dummy_feature_ch_first, axes=(1, 2, 0))
        dummy_input = tf.convert_to_tensor(dummy_feature_ch_last)
        dummy_input = tf.expand_dims(dummy_input, axis=0)
        _ = regressor(dummy_input, training=True)

        feature_height = 496
        feature_width = 718
        feature_ch = 4
        dummy_feature_ch_first = np.random.rand(feature_ch, feature_height, feature_width)
        dummy_feature_ch_last = np.transpose(dummy_feature_ch_first, axes=(1, 2, 0))
        dummy_input = tf.convert_to_tensor(dummy_feature_ch_last)
        dummy_input, _ = pad_img(dummy_input, height=feature_height, width=feature_width, divisor=32)
        dummy_input = tf.expand_dims(dummy_input, axis=0)
        print(dummy_input.shape)
        _ = fe(dummy_input, training=True)
        fe, classifier, regressor = load_torch_statedict(backbone=fe, mlp_classification=classifier,
                                                         mlp_regression=regressor, statedict_path=weight_path)

    return fe, classifier, regressor


if __name__ == '__main__':
    vgg13_pt = torchvision.models.vgg13(pretrained=False)

    fe = FeatureExtractor()

    dummy_feature_ch_first = np.random.rand(64 + 32, 256, 368)
    dummy_feature_ch_last = np.transpose(dummy_feature_ch_first, axes=(1, 2, 0))
    dummy_input = tf.convert_to_tensor(dummy_feature_ch_last)
    dummy_input = tf.expand_dims(dummy_input, axis=0)
    classifier = Classifier(filter_channels=[fe.out_ch, 512, 256, 128, fe.max_disp])
    classifier_output = classifier(dummy_input, training=True)

    dummy_feature_ch_first = np.random.rand(64 + 32 + 1, 256, 368)
    dummy_feature_ch_last = np.transpose(dummy_feature_ch_first, axes=(1, 2, 0))
    dummy_input = tf.convert_to_tensor(dummy_feature_ch_last)
    dummy_input = tf.expand_dims(dummy_input, axis=0)
    regressor = Regressor(filter_channels=[fe.out_ch + 1, 128, 64, 1])
    regression_output = regressor(dummy_input, training=True)

    feature_height = 496
    feature_width = 718
    feature_ch = 4
    dummy_feature_ch_first = np.random.rand(feature_ch, feature_height, feature_width)
    dummy_feature_ch_last = np.transpose(dummy_feature_ch_first, axes=(1, 2, 0))
    dummy_input = tf.convert_to_tensor(dummy_feature_ch_last)
    dummy_input, _ = pad_img(dummy_input, height=feature_height, width=feature_width, divisor=32)
    dummy_input = tf.expand_dims(dummy_input, axis=0)
    dummy_output = fe(dummy_input, training=True)

    # print(dummy_output)

    load_torch_statedict(backbone=fe, mlp_classification=classifier, mlp_regression=regressor)
