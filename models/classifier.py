import tensorflow as tf

from tensorflow import keras
from keras.layers import Layer, Conv1D, Softmax
from keras import Sequential
from keras import Model

from utils.activation_utils import Sine


class Classifier(Model):
    def __init__(self, filter_channels, use_softmax=True):
        super(Classifier, self).__init__()
        self.filters = []
        self.filter_channels = filter_channels
        self.use_softmax = use_softmax
        self.nfeat = filter_channels[0]
        self.layer_1 = Sequential(
            [
                Conv1D(filters=self.filter_channels[1], kernel_size=1, padding="same",
                                          kernel_initializer=tf.keras.initializers.HeUniform(),
                                          bias_initializer=tf.keras.initializers.Zeros()),
                Sine()
            ]
        )
        self.layer_2 = Sequential(
            [
                Conv1D(filters=self.filter_channels[2], kernel_size=1, padding="same",
                                          kernel_initializer=tf.keras.initializers.HeUniform(),
                                          bias_initializer=tf.keras.initializers.Zeros()),
                Sine()
            ]
        )
        self.layer_3 = Sequential(
            [
                Conv1D(filters=self.filter_channels[3], kernel_size=1, padding="same",
                                          kernel_initializer=tf.keras.initializers.HeUniform(),
                                          bias_initializer=tf.keras.initializers.Zeros()),
                Sine()
            ]
        )
        self.layer_4 = Sequential(
            [
                Conv1D(filters=self.filter_channels[4], kernel_size=1, padding="same",
                                          kernel_initializer=tf.keras.initializers.HeUniform(),
                                          bias_initializer=tf.keras.initializers.Zeros()),
            ]
        )
        # sofrmax over vector that goes from 0 all the way to max disparity
        # therefore, argmax of the vector gives the disparity value
        self.softmax = Softmax()

    def call(self, inputs, training=None):
        feat1 = self.layer_1(inputs)
        feat2 = self.layer_2(tf.concat([feat1, inputs], axis=-1))
        feat3 = self.layer_3(tf.concat([feat2, inputs], axis=-1))
        feat4 = self.layer_4(tf.concat([feat3, inputs], axis=-1))
        if self.use_softmax:
            probs = self.softmax(feat4)
            return probs, feat1, feat2, feat3, feat4
        else:
            return feat4
