
import numpy as np
import tensorflow as tf
import json
from src.lstm_attention_functions import *
import cv2
from src.dtd import rprop_attention_context_ab
from src.dtd import rprop_vgg16
from src.dtd import rprop_alpha
from src.model import build_model


def load_json(json_path):
    with open(json_path, 'r') as outfile:
        return json.load(outfile)


def save_json(data, output_json):
    with open(output_json,  'w') as outfile:
        json.dump(data, outfile, sort_keys=True, indent=4, ensure_ascii=False)


def transform_image(image_batch):
    img_trans = []
    for img in image_batch:
        img_trans.append(cv2.resize(img, (224, 224)))

    return img_trans


def vgg_demean(image_batch):
    vgg_mean = [103.939, 116.779, 123.68]
    img_trans = []
    for img in image_batch:
        blue, green, red = np.split(img, 3, 2)
        img_trans.append(np.concatenate([blue - vgg_mean[0], green - vgg_mean[1], red - vgg_mean[2]], 2))
    return img_trans


def vgg_addmean(image_batch):
    vgg_mean = [103.939, 116.779, 123.68]
    img_trans = []
    for img in image_batch:
        blue, green, red = np.split(img, 3, 2)
        img_trans.append(np.concatenate([blue + vgg_mean[0], green + vgg_mean[1], red + vgg_mean[2]], 2))
    return img_trans


def bgr_to_rgb(image_batch):
    img_trans = []
    for img in image_batch:
        blue, green, red = np.split(img, 3, 2)
        img_trans.append(np.concatenate([red, green, blue], 2))
    return img_trans


class CaptionGenerator(object):
    def __init__(
        self,
        hidden_state_dim=1024,
        pre_trained_cnn='vgg16',
        pre_trained_weights_path='data/vgg16.npy',
        layer_to_extract='conv5_3',
        path_to_bias='data/word_bias_init.npy',
        path_to_pretrained_model='',
        path_to_dictionary='data/dict_top_words_to_index.npy',
        emb_dim=512,
        word_frequency_threshold=10,
        max_sequence_len=16,
        attention_yn=True,
        attention_full=False,
        attention_context=True,
        start_of_sentence_token='sentence_start',
        end_of_sentence_token='sentence_end',
        ):

        self.hidden_state_dim = hidden_state_dim
        self.pre_trained_cnn = pre_trained_cnn
        self.pre_trained_weights_path = pre_trained_weights_path
        self.layer_to_extract = layer_to_extract
        self.path_to_pretrained_model = path_to_pretrained_model
        self.path_to_bias = path_to_bias
        self.path_to_dictionary = path_to_dictionary
        self.emb_dim = emb_dim
        self.word_frequency_threshold = word_frequency_threshold
        self.max_sequence_len = max_sequence_len
        self.attention_yn = attention_yn
        self.attention_full = attention_full
        self.attention_context = attention_context
        self.start_of_sentence_token = start_of_sentence_token
        self.end_of_sentence_token = end_of_sentence_token

        # load dictionary with top num_words to index
        self.dict_top_words_to_index = np.load(self.path_to_dictionary).item()

        # create dictionary index to word
        self.dict_index_to_top_words = {v: k for k, v in self.dict_top_words_to_index.iteritems()}
        self.num_words = len(self.dict_top_words_to_index)
        self.word_bias_init = np.load(self.path_to_bias)

        # load word embeddings
        self.tf_initialize_word_embeddings()

        # set tf placeholder
        self.batch_size = 1
        self.image_test = tf.placeholder(tf.float32, [self.batch_size, 224, 224, 3])

        # build pre trained encoder CNN
        if self.pre_trained_cnn == 'vgg16':
            self.vgg16_npy_path = self.pre_trained_weights_path
            self.weights_dict = np.load(self.vgg16_npy_path, encoding='latin1').item()  # load pre-trained weights

    def run_dtd(self, outputs, cp, weights, img_batch, approach, alpha=2):
        # run DTD
        if approach == 'dtda':
            dtda_yn = True
            alpha_relevance = rprop_alpha(outputs)
            alpha = 1
        elif approach == 'dtdb':
            dtda_yn = False
            alpha_relevance = rprop_attention_context_ab(outputs, cp, weights, 16, 2)
            alpha = 1
        elif approach == 'dtdc':
            dtda_yn = False
            alpha_relevance = rprop_attention_context_ab(outputs, cp, weights, 16, alpha)
        else:
            raise 'No approach specified (dtda, dtdb, or dtdc)'

        relevances = rprop_vgg16(
            weights,
            img_batch,
            alpha_relevance,
            self.weights_dict,
            self.batch_size,
            self.max_sequence_len,
            dtda=dtda_yn,
            alpha=alpha,
            )

        return relevances

    def get_outputs(self, val_img_path):
        self.dict_params, caption_indices, a, al, cp, am = build_model(
            self.max_sequence_len,
            self.batch_size,
            self.image_test,
            None,
            self.weights_dict,
            self.layer_to_extract,
            self.hidden_state_dim,
            self.emb_dim,
            len(self.dict_top_words_to_index.keys()),
            self.word_bias_init,
            0,
            0,
            self.dict_top_words_to_index,
            self.start_of_sentence_token,
            self.embedding_matrix,
            0,
            )

        fetches = {
            'caption_indices': caption_indices,
            'a_': a,
            'al_': al,
            'am_': am,
            }

        weights = {
            'att_W': self.dict_params['att_W'],
            'att_b': self.dict_params['att_b'],
            'att_pixel_W': self.dict_params['att_pixel_W'],
            'att_pixel_b': self.dict_params['att_pixel_b'],
            }

        # restore model and run
        saver = tf.train.Saver()
        with tf.Session() as sess:

            checkpoint_latest = tf.train.latest_checkpoint(
                self.path_to_pretrained_model,
                )
            saver.restore(sess, checkpoint_latest)

            print('model restored from %s' % (self.path_to_pretrained_model))

            val_img_load = cv2.imread(val_img_path)

            val_img = transform_image([val_img_load])
            val_img = vgg_demean(val_img)

            results_ = sess.run(fetches, feed_dict={self.image_test: val_img})
            cp_, am_ = sess.run([cp, am], feed_dict={self.image_test: val_img})
            weights_ = sess.run(weights)

        return results_, cp_, weights_, val_img, am_

    def tf_initialize_word_embeddings(self):
        # TODO: check whether this is better way to initialize with zeros before restoring model and overwriting
        embedding_matrix_np = np.zeros(
            (len(self.dict_top_words_to_index.keys()), self.emb_dim),
            ).astype('float32')

        with tf.variable_scope('word_embeddings'):
                self.embedding_matrix = tf.Variable(
                    embedding_matrix_np, name='emb_matrix',
                    )


if __name__ == '__main__':
    model = CaptionGenerator()
