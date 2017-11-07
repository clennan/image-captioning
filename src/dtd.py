
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src.modules.sequential import Sequential
from src.modules.alphas import Alphas
from src.modules.convolution import Convolution
from src.modules.convolution_ab import Convolution_ab
from src.modules.proppool import PropPool

import tensorflow as tf
import numpy as np


def vgg16_dtd(alphas_, batch_size, vgg16_weights, dtda=True, alpha=2):
    return Sequential([
        Convolution(batch_size=batch_size,
                    initializer=vgg16_weights['conv1_1'],
                    first=True,
                    name='conv1_1_'),
        Convolution_ab(
                    batch_size=batch_size,
                    initializer=vgg16_weights['conv1_2'],
                    alpha=alpha,
                    name='conv1_2_'),
        PropPool(name='proppool1'),
        Convolution_ab(
                    batch_size=batch_size,
                    initializer=vgg16_weights['conv2_1'],
                    alpha=alpha,
                    name='conv2_1_'),
        Convolution_ab(
                    batch_size=batch_size,
                    initializer=vgg16_weights['conv2_2'],
                    alpha=alpha,
                    name='conv2_2_'),
        PropPool(name='proppool2'),
        Convolution_ab(
                    batch_size=batch_size,
                    initializer=vgg16_weights['conv3_1'],
                    alpha=alpha,
                    name='conv3_1_'),
        Convolution_ab(
                    batch_size=batch_size,
                    initializer=vgg16_weights['conv3_2'],
                    alpha=alpha,
                    name='conv3_2_'),
        Convolution_ab(
                    batch_size=batch_size,
                    initializer=vgg16_weights['conv3_3'],
                    alpha=alpha,
                    name='conv3_3_'),
        PropPool(name='proppool3'),
        Convolution_ab(
                    batch_size=batch_size,
                    initializer=vgg16_weights['conv4_1'],
                    alpha=alpha,
                    name='conv4_1_'),
        Convolution_ab(
                    batch_size=batch_size,
                    initializer=vgg16_weights['conv4_2'],
                    alpha=alpha,
                    name='conv4_2_'),
        Convolution_ab(
                    batch_size=batch_size,
                    initializer=vgg16_weights['conv4_3'],
                    alpha=alpha,
                    name='conv4_3_'),
        PropPool(name='proppool4'),
        Convolution_ab(
                    batch_size=batch_size,
                    initializer=vgg16_weights['conv5_1'],
                    alpha=alpha,
                    name='conv5_1_'),
        Convolution_ab(
                    batch_size=batch_size,
                    initializer=vgg16_weights['conv5_2'],
                    alpha=alpha,
                    name='conv5_2_'),
        Convolution_ab(
                    batch_size=batch_size,
                    initializer=vgg16_weights['conv5_3'],
                    alpha=alpha,
                    name='conv5_3_'),
        Alphas(alphas_, dtda)])


def fprop_next_linear_alphabeta(x, f, w, b, alpha=2):
    x = x+1e-9
    beta = alpha - 1
    V_pos = np.maximum(1e-9, w)
    V_neg = np.minimum(-1e-9, w)
    Z_pos = np.dot(x, V_pos) + 1e-9 + np.maximum(b, 0) / 2
    Z_neg = np.dot(x, V_neg) + 1e-9 + np.maximum(b, 0) / 2
    S_pos = alpha * f/Z_pos
    S_neg = -beta * f/Z_neg
    C = np.dot(S_pos, V_pos.T) + np.dot(S_neg, V_neg.T)
    F = x*C
    return F


def rprop_alpha(outputs):
    relevance = []
    for r in outputs['a_']:
        new_rel = np.stack([r.reshape([14, 14])]*512, axis=-1)
        assert new_rel.shape == (14, 14, 512)
        relevance.append(np.expand_dims(new_rel, axis=0))
    return relevance


def rprop_attention_context_ab(outputs, cp_, weights, max_seq_len, alpha):
    rel_al_ = np.squeeze(outputs['a_']) * np.maximum(0, np.squeeze(outputs['al_']))
    assert rel_al_.shape == (16, 196)
    rel_al_ = np.expand_dims(rel_al_, axis=-1)  # (16, 196, 1)

    cp_ = np.maximum(0, np.squeeze(cp_))  # (196, 512)
    cp_ = np.squeeze(cp_)
    assert cp_.shape == (196, 512)

    print('using alpha beta')

    rel_att_ = []
    for c, _ in enumerate(rel_al_):
        rel_att_.append(
            fprop_next_linear_alphabeta(
                cp_,
                rel_al_[c],
                weights['att_W'],
                weights['att_b'],
                alpha,
                ))

    relevance = []
    for r in rel_att_:
        relevance.append(r.reshape([196, 512]))

    am_ = np.squeeze(outputs['am_'])

    rel_att_pixel_ = np.array(relevance)
    assert rel_att_pixel_.shape == (16, 196, 512)

    rel_pixel_ = []
    for r, rel in enumerate(rel_att_pixel_):
        rel_pixel_.append(
            fprop_next_linear_alphabeta(
                am_,
                rel,
                weights['att_pixel_W'],
                weights['att_pixel_b'],
                alpha,
                ))

    relevance = []
    for r in rel_pixel_:
        relevance.append(r.reshape([-1, 14, 14, 512]))

    return relevance


def rprop_vgg16(
    weights,
    images,
    alpha_relevance,
    vgg16_weights,
    batch_size,
    max_sequence_len,
    dtda=False,
    alpha=2,
    ):

    with tf.Session() as sess:
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
            alphas_ = tf.placeholder(tf.float32, [batch_size, 14, 14, 512])

        with tf.variable_scope('model'):
            cnn = vgg16_dtd(alphas_, batch_size, vgg16_weights, dtda, alpha)
            output = cnn.forward(x)

        with tf.variable_scope('relevance'):
            heatmaps = cnn.lrp(output)

        tf.global_variables_initializer().run()
        relevances = []
        for w in range(max_sequence_len):
            relevances.append(
                sess.run(heatmaps,
                         feed_dict={x: images, alphas_: alpha_relevance[w]},
                         ),
                )

    return relevances
