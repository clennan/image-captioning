
import os
import glob
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import json
import operator
from collections import Counter
from collections import defaultdict
import re
import argparse


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_image(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg, feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _process_image(filename, coder):
    """Process a single image file.
    Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'r') as f:
        image_data = f.read()

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


def preprocess_image(image_buffer, size):
    '''
    Args:
        TF symbolic image or images of dimensions
        [height, width, channels] or [batch, height, width, channels]
    '''
    image_jpeg = decode_jpeg(image_buffer)
    image_resize = tf.image.resize_images(image_jpeg, size)
    dims = image_resize.get_shape()
    num_dims = len(dims)

    image_batch_not_scaled = image_resize * 255.0

    # convert from RGB to BGR
    # split into 3 tenors along last dimension
    red, green, blue = tf.split(image_batch_not_scaled, 3, num_dims-1)
    assert red.get_shape().as_list()[num_dims-3:] == [224, 224, 1]
    assert green.get_shape().as_list()[num_dims-3:] == [224, 224, 1]
    assert blue.get_shape().as_list()[num_dims-3:] == [224, 224, 1]
    # de-mean data https://gist.github.com/ksimonyan/211839e770f7b538e2d8#description
    vgg_mean = [103.939, 116.779, 123.68]
    image_batch_demeaned = tf.concat(
        [blue - vgg_mean[0], green - vgg_mean[1], red - vgg_mean[2]],
        num_dims-1,
        )

    assert image_batch_demeaned.get_shape().as_list()[num_dims-3:] == [224, 224, 3]
    return image_batch_demeaned


def decode_jpeg(image_buffer, scope=None):
    """https://github.com/tensorflow/models/blob/master/inception/inception/image_processing.py
    Decode a JPEG string into one 3-D float image Tensor.
    Args:
    image_buffer: scalar string Tensor.
    scope: Optional scope for op_scope.
    Returns:
    3-D float Tensor with values ranging from [0, 1).
    """
    with tf.name_scope(scope, 'decode_jpeg', [image_buffer]):
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height and width
        # that is set dynamically by decode_jpeg. In other words, the height
        # and width of image is unknown at compile-time.
        image = tf.image.decode_jpeg(image_buffer, channels=3)

        # After this point, all image pixels reside in [0,1)
        # until the very end, when they're rescaled to (-1, 1).  The various
        # adjust_* ops all require this range for dtype float.
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image


def serialize_data(
    path_to_image_folder,
    path_to_captions_file,
    path_to_save,
    frequency_threshold,
    max_sequence_len,
    coder,
    train,
    start_of_sentence_token='sentence_start',
    end_of_sentence_token='sentence_end',
    ):

    if train:
        tfrecords_name = 'train.tfrecords'
    else:
        tfrecords_name = 'val.tfrecords'

    writer = tf.python_io.TFRecordWriter(os.path.join(path_to_save, tfrecords_name))

    # build dictionary mapping image id to caption, store captions in list as every image has multiple captions
    dict_imageid_to_caption = defaultdict(list)

    with open(path_to_captions_file, 'r') as f:
        read_data = f.read()

    read_data = json.loads(read_data)

    for ann in read_data['annotations']:
        caption = text_preprocessing(ann['caption'])
        # start and end of sentence tokens
        dict_imageid_to_caption[ann['image_id']].append(
            [start_of_sentence_token] + caption + [end_of_sentence_token])

    if train:
        # build vocabulary from dict_imageid_to_caption
        vocabulary = build_vocabulary_with_word_count(dict_imageid_to_caption)
        bias_init, vocabulary_top_list = keep_top_words_and_bias(vocabulary, frequency_threshold)

        # create dictionary with top num_words to index
        dict_top_words_to_index = {}
        for i in range(len(vocabulary_top_list)):
            dict_top_words_to_index[vocabulary_top_list[i]] = i+1
        # reserve 0 index as placeholder for padded sequences
        dict_top_words_to_index['zero_pad_token'] = 0

        # save dictionary
        save_path = os.path.join(path_to_save, 'dict_top_words_to_index.npy')
        np.save(save_path, dict_top_words_to_index)

        # save bias initialization
        save_path = os.path.join(path_to_save, 'word_bias_init.npy')
        np.save(save_path, bias_init)

    else:
        path_to_dictionary = os.path.join(path_to_save, 'dict_top_words_to_index.npy')
        dict_top_words_to_index = np.load(path_to_dictionary).item()

    # get list of images to load
    image_files = glob.glob(os.path.join(path_to_image_folder, '*.jpg'))
    image_file_stem = image_files[0][:-16]

    for ann in tqdm(read_data['annotations']):
        caption = text_preprocessing(ann['caption'])
        image_path = os.path.join(
            path_to_image_folder,
            image_file_stem+str(ann['image_id']).zfill(12)+'.jpg',
            )
        image, height, width = _process_image(image_path, coder)
        image_id = int(image_path[-16:-4].lstrip('0'))

        # create sequence of indices based on dict_top_words_to_index
        # ignores words in caption that are not found in dict_top_words_to_index
        label_sequence = [dict_top_words_to_index.get(word) for word in caption if dict_top_words_to_index.get(word)]

        # cut sequence to max_sequence_len and add eos_toke
        # if sequence is shorter than max_sequence_len, fill with 0s
        label_sequence = label_sequence[:max_sequence_len - 1]
        label_sequence += [0]*(max_sequence_len - len(label_sequence))

        # construct the example proto object
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'label': tf.train.Feature(
                        # bytes_list=tf.train.BytesList(value=[label_sequence])),
                        int64_list=tf.train.Int64List(value=label_sequence)),
                    'image': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[image])),
                    'id': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[image_id])),
                        }
                        ))
        # use the proto object to serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to disk
        writer.write(serialized)
    print('data file saved under %s' % (os.path.join(path_to_save, tfrecords_name)))


def read_and_decode_example(
    filename,
    max_sequence_len,
    num_epochs,
    size=[224, 224],
    ):

    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=num_epochs,
        )
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    feature_map = {
      'image': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
      'id': tf.FixedLenFeature([1], dtype=tf.int64),
      # 'label': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
      'label': tf.FixedLenFeature([max_sequence_len], dtype=tf.int64),
      }

    features = tf.parse_single_example(serialized_example, feature_map)
    processed_image = preprocess_image(features['image'], size)

    return processed_image, features['label'], features['id']


def text_preprocessing(caption):
    # split string into words and remove punctuations
    caption = re.findall('\w+', caption)
    caption = [word.lower() for word in caption]  # lower case
    return caption


def build_vocabulary_with_word_count(dictionary):
    vocabulary = Counter()
    for _, captions_list in dictionary.items():
        for caption in captions_list:
            vocabulary.update(caption)
    return vocabulary


def keep_top_words_and_bias(
    vocabulary_with_word_count,
    frequency_threshold,
    ):
    # sort vocabulary by frequencies
    vocabulary_sorted = sorted(
        vocabulary_with_word_count.items(),
        key=operator.itemgetter(1),
        reverse=True,
        )
    temp = [(vocabulary_sorted[i][0], vocabulary_sorted[i][1]) for i in range(len(vocabulary_sorted)) if vocabulary_sorted[i][1] >= frequency_threshold]

    bias = np.array([1.0]+[1.0*temp[i][1] for i, _ in enumerate(temp)])
    bias /= np.sum(bias)
    bias = np.log(bias)
    bias -= np.max(bias)

    return bias, [temp[i][0] for i, _ in enumerate(temp)]


if __name__ == "__main__":
    coder = ImageCoder()
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
      '--images-dir',
      help='',
      required=True
      )

    parser.add_argument(
      '--label-file',
      help='',
      required=True
      )

    parser.add_argument(
      '--save-dir',
      help='',
      required=True
      )

    parser.add_argument(
      '--train',
      help='if False, dict_top_words_to_index.npy must be in save-dir',
      required=True
      )

    args = parser.parse_args()
    arguments = args.__dict__

    print('serializing data...')
    serialize_data(
        arguments['images_dir'],
        arguments['label_file'],
        arguments['save_dir'],
        10,
        16,
        coder,
        arguments['train'].lower() == 'true',
        )
