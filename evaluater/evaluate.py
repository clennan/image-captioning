
import os
import json
import numpy as np
import tensorflow as tf
import argparse
from StringIO import StringIO
from tensorflow.python.lib.io import file_io
from src.model import build_model
from src.convert_to_tfrecords import read_and_decode_example


def save_json(data, output_json):
    with open(output_json,  'w') as outfile:
        json.dump(data, outfile, sort_keys=True, indent=4, ensure_ascii=False)


def convert_index_sequence_to_sentence(
    list_of_sequences,
    dict_idx_to_word,
    end_of_sentence_token=None,
    ):

    list_of_sentences = []
    for s, sequence in enumerate(list_of_sequences):
        temp = [dict_idx_to_word[i] for i in sequence]
        if end_of_sentence_token:
            try:
                eos_index = temp.index(end_of_sentence_token)
                temp = temp[:eos_index+1]
            except ValueError:
                temp = temp
        list_of_sentences.append(' '.join(temp))
    return list_of_sentences


class Eval(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        # load dictionary with top num_words to index
        tmp = StringIO(file_io.read_file_to_string(self.dict_file))
        self.dict_top_words_to_index = np.load(tmp).item()

        # create dictionary index to word
        self.dict_index_to_top_words = {v: k for k, v in self.dict_top_words_to_index.iteritems()}
        self.num_words = len(self.dict_top_words_to_index)

        # load word embeddings
        self.tf_initialize_word_embeddings()

        # test set: initialize .tfrecords reader and tf batch variables
        print('initialize .tfrecords from: '+self.test_file)
        self.image, self.label, self.id = read_and_decode_example(
            self.test_file, self.max_sequence_len, num_epochs=1,
            )

        self.image_batch, self.label_batch, self.ids_batch = tf.train.batch(
            [self.image, self.label, self.id],
            batch_size=self.batch_size,
            allow_smaller_final_batch=False,
            enqueue_many=False,
            )

        # load pre-trained CNN weights
        tmp = StringIO(file_io.read_file_to_string(self.weights_file))
        self.weights_dict = np.load(tmp, encoding='latin1').item()

        # run evaluation
        self.eval()

    def tf_initialize_word_embeddings(self):
        with tf.variable_scope('word_embeddings'):
            self.embedding_matrix = tf.get_variable(
                'emb_matrix', initializer=tf.random_uniform(
                    shape=[self.num_words, self.emb_dim]), dtype=tf.float32,
                    )

    def eval(self):
        _, generated_word_indices, _, _, _, _ = build_model(
            self.max_sequence_len,
            self.batch_size,
            self.image_batch,
            None,
            self.weights_dict,
            self.layer_to_extract,
            self.hidden_state_dim,
            self.emb_dim,
            self.num_words,
            None,
            0,
            0,
            self.dict_top_words_to_index,
            self.start_of_sentence_token,
            self.embedding_matrix,
            0
            )

        saver = tf.train.Saver(var_list=tf.trainable_variables())
        output_json = []
        with tf.Session() as sess:
            checkpoint_latest = tf.train.latest_checkpoint(self.model_dir)
            saver.restore(sess, checkpoint_latest)
            print('model restored from %s' % (self.model_dir))

            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess=sess)

            i = 0
            try:
                print('start propagating val images...')
                while not coord.should_stop():
                    i += 1
                    images, labels, image_ids, generated_captions = sess.run(
                        [self.image_batch, self.label_batch, self.ids_batch, generated_word_indices],
                        )

                    generated_captions = zip(*generated_captions)

                    for c, caption in enumerate(generated_captions):
                        caption_ = convert_index_sequence_to_sentence(
                            [caption], self.dict_index_to_top_words,
                            self.end_of_sentence_token,
                            )

                        output_json.append({
                            'image_id': image_ids[c][0],
                            'caption': caption_[0],
                            })

                    print('batch %s' % (i))

            except tf.errors.OutOfRangeError:
                print('evaluation finished, input queue is empty')
            finally:
                coord.request_stop()

            coord.request_stop()
            sess.close()

            json_name = 'captions_'+os.path.basename(checkpoint_latest)+'.json'
            save_json(output_json, os.path.join(self.job_dir, json_name))
            print('generated captions saved to %s' % os.path.join(self.job_dir, json_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
      '--test-file',
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
      '--dict-file',
      help='',
      required=True
      )

    parser.add_argument(
      '--model-dir',
      help='',
      required=True
      )

    args = parser.parse_args()
    arguments = args.__dict__
    job_dir = arguments['job_dir']

    params = {
        'batch_size': 25,
        'min_after_dequeue': 500,
        'learning_rate': 0.001,
        'hidden_state_dim': 1024,
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

    Eval(**params)
