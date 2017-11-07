
import tensorflow as tf
from module import Module


def weights_pretrained(weights, name=''):
    return tf.get_variable(name+'/weights', initializer=weights)


def biases_pretrained(bias, name=''):
    return tf.get_variable(name+'/biases', initializer=bias)


def fprop_next_conv(x, f, w, b, batch_size, alpha=2):
    x = x+1e-9

    beta = alpha-1
    # batch_size = tf.shape(x)[0]
    x_dims = x.get_shape().as_list()
    deconv_shape = tf.stack([batch_size]+x_dims[1:])

    v_p = tf.maximum(w, 1e-9)
    v_n = tf.minimum(-w, -1e-9)
    z_p = tf.nn.conv2d(x, v_p, strides=[1, 1, 1, 1], padding='SAME') + tf.maximum(b, 0) / 2
    z_n = tf.nn.conv2d(x, v_n, strides=[1, 1, 1, 1], padding='SAME') + tf.maximum(b, 0) / 2
    s_p = alpha * tf.divide(f, z_p)
    s_n = -beta * tf.divide(f, z_n)
    c = tf.nn.conv2d_transpose(s_p, v_p, deconv_shape, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d_transpose(s_n, v_n, deconv_shape, strides=[1, 1, 1, 1], padding='SAME')

    return x * c


def fprop_first_conv(x, f, w, b, batch_size, lowest, highest):
    x_dims = x.get_shape().as_list()
    deconv_shape = tf.stack([batch_size]+x_dims[1:])

    v = tf.nn.relu(w)
    u = tf.minimum(w, 0)
    l = tf.cast(x * 0 + lowest, dtype=tf.float32)
    h = tf.cast(x * 0 + highest, dtype=tf.float32)

    z = (tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
         - tf.nn.conv2d(l, v, strides=[1, 1, 1, 1], padding='SAME')
         - tf.nn.conv2d(h, u, strides=[1, 1, 1, 1], padding='SAME')) + tf.maximum(b, 0)

    s = tf.divide(f, z)

    f = (x * tf.nn.conv2d_transpose(s, w, deconv_shape, strides=[1, 1, 1, 1], padding='SAME')
         - l * tf.nn.conv2d_transpose(s, v, deconv_shape, strides=[1, 1, 1, 1], padding='SAME')
         - h * tf.nn.conv2d_transpose(s, u, deconv_shape, strides=[1, 1, 1, 1], padding='SAME'))

    return f


class Convolution_ab(Module):

    def __init__(self,
                 batch_size=None,
                 initializer=None,
                 alpha=2,
                 name='conv_ab',
                 first=False,
                 lowest=-200,
                 highest=200):

        self.name = name
        self.alpha=alpha
        self.batch_size = batch_size
        self.initializer = initializer
        self.first = first
        self.lowest = lowest
        self.highest = highest

    def forward(self, input_tensor):
        with tf.variable_scope(self.name):
            self.input_tensor = input_tensor
            self.weights = weights_pretrained(self.initializer[0])
            self.biases = biases_pretrained(self.initializer[1])

            conv = tf.nn.conv2d(
                self.input_tensor, self.weights, strides=[1, 1, 1, 1],
                padding='SAME',
                )

            self.activations = tf.nn.relu(tf.nn.bias_add(conv, self.biases))

        return self.activations

    def lrp(self, R):
        if self.first:
            return fprop_first_conv(
                self.input_tensor, R, self.weights, self.biases,
                self.batch_size, self.lowest, self.highest,
                )
        else:
            return fprop_next_conv(
                self.input_tensor, R, self.weights, self.biases,
                self.batch_size, self.alpha,
                )
