import tensorflow as tf
from keras.layers import Layer


class Sine(Layer):
    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def call(self, inputs):
        return tf.math.sin(self.w0 * inputs)