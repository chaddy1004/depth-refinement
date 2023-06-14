import tensorflow as tf

from tensorflow import keras
from keras.layers import Layer, Conv1D, Softmax, Activation
from keras import Model
from keras import Sequential
from keras import initializers

from utils.activation_utils import Sine


class Regressor(Model):
    def __init__(self, filter_channels, use_softmax=True):
        super(Regressor, self).__init__()
        self.filters = []
        self.filter_channels = filter_channels
        self.n_feat = filter_channels[0]

        # layer1
        self.layer_1 = Sequential()
        self.layer_1.add(
            Conv1D(filters=filter_channels[1], kernel_size=1, kernel_initializer=tf.keras.initializers.HeUniform(),
                   bias_initializer=tf.keras.initializers.Zeros()))
        self.layer_1.add(Sine())

        # layer2
        self.layer_2 = Sequential()
        self.layer_2.add(
            Conv1D(filters=filter_channels[2], kernel_size=1, kernel_initializer=tf.keras.initializers.HeUniform(),
                   bias_initializer=tf.keras.initializers.Zeros()))
        self.layer_2.add(Sine())

        # layer3
        self.layer_3 = Conv1D(filters=1, kernel_size=1, kernel_initializer=tf.keras.initializers.HeUniform(),
                              bias_initializer=tf.keras.initializers.Zeros())
        self.activation = Activation("tanh")

    def call(self, input_feature, training=None):
        x = self.layer_1(input_feature)
        x = self.layer_2(tf.concat([x, input_feature], axis=-1))
        x = self.layer_3(tf.concat([x, input_feature], axis=-1))
        offsets = self.activation(x)
        return offsets
