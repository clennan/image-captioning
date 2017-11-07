
import tensorflow as tf
from module import Module


class PropPool(Module):

    def __init__(self, pool_size=2, name='proppool'):

        self.name = name
        self.pool_size = pool_size

    def forward(self, input_tensor):
        with tf.variable_scope(self.name):
            self.input_tensor = input_tensor
            self.activations = tf.nn.max_pool(
                self.input_tensor, ksize=[1, self.pool_size, self.pool_size, 1], strides=[1, self.pool_size, self.pool_size, 1],
                padding='SAME',
                )

            # save activations from avg pooling for lrp
            self.activations_lrp = tf.nn.avg_pool(
                self.input_tensor, ksize=[1, self.pool_size, self.pool_size, 1], strides=[1, self.pool_size, self.pool_size, 1],
                padding='SAME',
                )

        return self.activations

    def lrp(self, R):
        # print('using prop pooling')
        self.R = R
        Y = self.activations_lrp * self.pool_size * self.pool_size
        R_y = R / (Y+1e-9)
        R_y_up = tf.contrib.keras.layers.UpSampling2D(size=self.pool_size)(R_y)
        return self.input_tensor * R_y_up
