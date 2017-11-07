
import numpy as np
import tensorflow as tf
import argparse
from StringIO import StringIO
from tensorflow.python.lib.io import file_io
from src.model import build_model
from tools.convert_to_tfrecords import read_and_decode_example


class Train(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        # load dictionary with top num_words to index
        tmp = StringIO(file_io.read_file_to_string(self.dict_file))
        self.dict_top_words_to_index = np.load(tmp).item()

        # create dictionary index to word
        self.dict_index_to_top_words = {v: k for k, v in self.dict_top_words_to_index.iteritems()}

        self.num_words = len(self.dict_top_words_to_index)

        tmp = StringIO(file_io.read_file_to_string(self.bias_file))
        self.word_bias_init = np.load(tmp)

        # load word embeddings
        self.tf_initialize_word_embeddings()
        print('word embeddings randomly initialized')

        # training set: initialize .tfrecords reader and tf batch variables
        print('initialize .tfrecords from: '+self.train_file)
        self.image_train, self.label_train, self.id_train = read_and_decode_example(
            self.train_file, self.max_sequence_len, self.num_epochs,
            )

        self.image_batch_train, self.label_batch_train = tf.train.shuffle_batch(
            [self.image_train, self.label_train],
            batch_size=self.batch_size,
            capacity=self.min_after_dequeue + 3 * self.batch_size,
            min_after_dequeue=self.min_after_dequeue,
            )

        # load pre-trained CNN weights
        tmp = StringIO(file_io.read_file_to_string(self.weights_file))
        self.weights_dict = np.load(tmp, encoding='latin1').item()

        # run training
        self.train()

    def tf_initialize_word_embeddings(self):
        with tf.variable_scope('word_embeddings'):
            self.embedding_matrix = tf.get_variable(
                'emb_matrix', initializer=tf.random_uniform(
                    shape=[self.num_words, self.emb_dim]), dtype=tf.float32,
                    )

    def train(self):
        batch_loss, _ = build_model(
            self.max_sequence_len,
            self.batch_size,
            self.image_batch_train,
            self.label_batch_train,
            self.weights_dict,
            self.layer_to_extract,
            self.hidden_state_dim,
            self.emb_dim,
            self.num_words,
            self.word_bias_init,
            self.dropout_lstm_init,
            self.dropout_lstm,
            self.dict_top_words_to_index,
            self.start_of_sentence_token,
            self.embedding_matrix,
            self.alpha_regularization,
            )

        lstm_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='lstm_and_attention',
            )

        word_emb_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='word_embeddings',
            )

        # define separate train steps for lstm and soft attention scope
        # suggested method by mrry http://stackoverflow.com/questions/35298326/freeze-some-variables-scopes-in-tensorflow-stop-gradient-vs-passing-variables
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(
            batch_loss, var_list=lstm_vars,
            )

        train_step2 = tf.train.AdamOptimizer(self.learning_rate).minimize(
            batch_loss, var_list=word_emb_vars,
            )

        with tf.Session() as sess:
            saver = tf.train.Saver(max_to_keep=20)
            init = tf.group(tf.local_variables_initializer(),
                            tf.global_variables_initializer())
            sess.run(init)

            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)

            merged_train = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.job_dir + '/train')

            i = 0
            try:
                print('start training...')
                while not coord.should_stop():
                    i += 1
                    epoch = 1 + (self.batch_size * i) / (5*82000)
                    summary, loss, _, _ = sess.run(
                        [merged_train, batch_loss, train_step, train_step2],
                        )
                    train_writer.add_summary(summary, i)
                    print('train set: epoch %s batch %s: %s' % (epoch, i, loss))

                    if i % 100 == 0:
                        saver.save(sess, self.job_dir+'/model.ckpt', i)

            except tf.errors.OutOfRangeError:
                print('training finished, input queue is empty')
            finally:
                coord.request_stop()

            coord.request_stop()
            sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
      '--train-file',
      help='',
      required=True
      )

    parser.add_argument(
      '--job-dir',
      help='',
      required=True
      )

    parser.add_argument(
      '--weights-file',
      help='',
      required=True
      )

    parser.add_argument(
      '--bias-file',
      help='',
      required=True
      )

    parser.add_argument(
      '--dict-file',
      help='',
      required=True
      )

    args = parser.parse_args()
    arguments = args.__dict__
    job_dir = arguments['job_dir']

    params = {
        'batch_size': 32,
        'min_after_dequeue': 500,
        'num_epochs': 5,
        'learning_rate': 0.001,
        'dropout_lstm': 0.75,
        'dropout_lstm_init': 0,
        'alpha_regularization': 1.0,
        'hidden_state_dim': 1024,
        'weights_file': '',
        'layer_to_extract': 'conv5_3',
        'emb_dim': 512,
        'word_frequency_threshold': 10,
        'max_sequence_len': 16,
        'start_of_sentence_token': 'sentence_start',
        'end_of_sentence_token': 'sentence_end',
        }

    # update parameters with arguments provided in function call
    for key, value in arguments.items():
        params[key] = value

    Train(**params)
