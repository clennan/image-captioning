
import tensorflow as tf
from lstm_attention_functions import soft_attention_initializations
from lstm_attention_functions import lstm_initializations
from lstm_attention_functions import initial_lstm_states
from lstm_attention_functions import soft_attention_step
from lstm_attention_functions import lstm_step


def conv_layer_pre_trained(input, weights_dict, layer_name):
    # Create variable named "weights" initialized with pre-trained weights
    weights = tf.get_variable(
        "weights", initializer=weights_dict[layer_name][0],
        )
    # Create variable named "biases" initialized with pre-trained weights
    biases = tf.get_variable(
        "biases", initializer=weights_dict[layer_name][1],
        )
    conv = tf.nn.conv2d(
        input, weights, strides=[1, 1, 1, 1], padding='SAME',
        )
    return tf.nn.relu(conv + biases)


def fc_layer_relu_pre_trained(input, weights_dict, layer_name):
    shape = input.get_shape().as_list()
    dim = 1
    for d in shape[1:]:
        dim *= d
    x = tf.reshape(input, [-1, dim])

    weights = tf.get_variable(
        "weights", initializer=weights_dict[layer_name][0],
        )
    biases = tf.get_variable(
        "biases", initializer=weights_dict[layer_name][1],
        )
    return tf.nn.relu(tf.matmul(x, weights) + biases)


def max_pool(input):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


def build_vgg16(images, weights_dict, layer_to_extract):
    # build model graph
    with tf.variable_scope('vgg16'):
        with tf.variable_scope('conv1_1'):
            conv1_1 = conv_layer_pre_trained(images, weights_dict, "conv1_1")
        with tf.variable_scope('conv1_2'):
            conv1_2 = conv_layer_pre_trained(conv1_1, weights_dict, "conv1_2")
        pool1 = max_pool(conv1_2)

        with tf.variable_scope('conv2_1'):
            conv2_1 = conv_layer_pre_trained(pool1, weights_dict, "conv2_1")
        with tf.variable_scope('conv2_2'):
            conv2_2 = conv_layer_pre_trained(conv2_1, weights_dict, "conv2_2")
        pool2 = max_pool(conv2_2)

        with tf.variable_scope('conv3_1'):
            conv3_1 = conv_layer_pre_trained(pool2, weights_dict, "conv3_1")
        with tf.variable_scope('conv3_2'):
            conv3_2 = conv_layer_pre_trained(conv3_1, weights_dict, "conv3_2")
        with tf.variable_scope('conv3_3'):
            conv3_3 = conv_layer_pre_trained(conv3_2, weights_dict, "conv3_3")
        pool3 = max_pool(conv3_3)

        with tf.variable_scope('conv4_1'):
            conv4_1 = conv_layer_pre_trained(pool3, weights_dict, "conv4_1")
        with tf.variable_scope('conv4_2'):
            conv4_2 = conv_layer_pre_trained(conv4_1, weights_dict, "conv4_2")
        with tf.variable_scope('conv4_3'):
            conv4_3 = conv_layer_pre_trained(conv4_2, weights_dict, "conv4_3")
        pool4 = max_pool(conv4_3)

        with tf.variable_scope('conv5_1'):
            conv5_1 = conv_layer_pre_trained(pool4, weights_dict, "conv5_1")
        with tf.variable_scope('conv5_2'):
            conv5_2 = conv_layer_pre_trained(conv5_1, weights_dict, "conv5_2")
        with tf.variable_scope('conv5_3'):
            conv5_3 = conv_layer_pre_trained(conv5_2, weights_dict, "conv5_3")

    return locals()[layer_to_extract]


def reshape_activation_maps(activation_maps):
    dims = activation_maps.get_shape().as_list()
    num_pixels = dims[1] * dims[2]  # 14 * 14 = 196
    num_filters = dims[-1]  # 512
    # [batch_size, 196, 512]
    activation_maps_flat = tf.reshape(activation_maps,
                                      [-1, num_pixels, num_filters])
    return activation_maps_flat, num_pixels, num_filters


def build_model(
    max_sequence_len,
    batch_size,
    input_images,
    input_labels,
    weights_dict,
    layer_to_extract,
    hidden_state_dim,
    emb_dim,
    num_words,
    word_bias_init,
    dropout_lstm_init,
    dropout_lstm,
    dict_top_words_to_index,
    start_of_sentence_token,
    embedding_matrix,
    alpha_regularization,
    ):

    if input_labels is not None:
        # create mask for entire batch
        mask_batch = tf.not_equal(input_labels, 0)

        # split labels from current batch
        # returns list with self.max_sequence_len number of elements
        # each element has [batch_size, 1]
        label_batch_split = tf.split(input_labels, int(max_sequence_len), 1)

    # get activation maps from encoder CNN
    with tf.variable_scope('cnn_encoder'):
        activation_maps = build_vgg16(input_images, weights_dict,
                                      layer_to_extract)

    # flatten activation maps and infer dimensions
    activation_maps_flat, num_pixels, num_filters = reshape_activation_maps(activation_maps)

    # initialize soft attention and lstm cell
    dict_params = {}
    with tf.variable_scope('lstm_and_attention'):
        dict_params = soft_attention_initializations(
            dict_params, num_filters, hidden_state_dim,
            )

        dict_params = lstm_initializations(
            dict_params,
            emb_dim,
            hidden_state_dim,
            num_filters,
            num_words,
            word_bias_init=word_bias_init,
            )

    alpha_logits_list = []
    alpha_list = []
    mask_list = []
    generated_word_indices = []
    for step in range(max_sequence_len):
        if input_labels is not None:
            label_current = tf.reshape(label_batch_split[step], [-1])

            # generate mask for cross entropy loss
            # if sequence has ended, i.e. label is 0, we don't want to add it to loss
            mask_current = tf.not_equal(label_current, 0)
            mask_list.append(mask_current)

            # convert labels to one-hot format
            # i.e. [num_words] zero vector with 1 at word index
            labels_one_hot = tf.one_hot(label_current, num_words)

        if step == 0:
            # initialize loss
            loss = tf.constant(0.0)

            # initialize LSTM hidden and memory state
            hidden_state_prev, memory_prev = initial_lstm_states(
                dict_params, tf.reduce_mean(activation_maps_flat, 1), dropout_lstm_init,
                )

            # project activations from CNN only once at step 0
            # see line 355 https://github.com/kelvinxu/arctic-captions/blob/master/capgen.py
            # in order to apply self.att_pixel_W to each pixel, activations are stacked to
            # [batch_size * 196, 512]
            activation_maps_stacked = tf.reshape(activation_maps_flat, [-1, num_filters])

            # project context, dims unchanged [batch_size * 196, 512]
            context_projected = tf.matmul(
                activation_maps_stacked, dict_params['att_pixel_W']
                ) + dict_params['att_pixel_b']

            # reshape back to [batch_size, 196, 512]
            context_projected = tf.reshape(context_projected, [-1, num_pixels, num_filters])

            # initialize word input at step 0 with start of sentence token
            start_of_sentence_idx = dict_top_words_to_index[start_of_sentence_token]

            embedded_word_prev = tf.nn.embedding_lookup(
                embedding_matrix, start_of_sentence_idx,
                )

            embedded_word_prev = tf.tile(
                tf.reshape(embedded_word_prev, [1, emb_dim]),
                [batch_size, 1],
                )

        else:
            # update memory and hidden state
            memory_prev = memory
            hidden_state_prev = hidden_state

            # get previous and current index from sequence labels
            if input_labels is not None:
                # take t-1 word for producing h_t and c_t
                label_prev = label_batch_split[step-1]
                label_prev = tf.reshape(label_prev, [-1])
            else:
                # if evaluating, take previous generated word
                label_prev = generated_word_indices[-1]

            # look up previous word in embedding matrix
            embedded_word_prev = tf.nn.embedding_lookup(embedding_matrix,
                                                        label_prev)

        # update attention weights and apply to context
        weighted_context, alpha, alpha_logits = soft_attention_step(
            dict_params, context_projected, hidden_state_prev,
            activation_maps_flat, num_filters, num_pixels,
            )

        alpha_list.append(alpha)
        alpha_logits_list.append(alpha_logits)

        # run lstm step
        logits, hidden_state, memory = lstm_step(
            dict_params,
            embedded_word_prev,
            hidden_state_prev,
            memory_prev,
            dropout_lstm,
            weighted_context,
            )

        # store most probable word
        most_probable_word = tf.argmax(logits, 1)
        # values, indices = tf.nn.top_k(logits, 2)
        # most_probable_word = indices[:, 1]
        generated_word_indices.append(most_probable_word)

        if input_labels is not None:
            # calculate cross entropy loss for each sample in batch
            cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=labels_one_hot, logits=logits,
                )

            masked_cross_entropy_loss = tf.boolean_mask(cross_entropy_loss, mask_current)

            current_loss = tf.reduce_sum(masked_cross_entropy_loss)
            loss += current_loss

    if input_labels is None:
        return dict_params, generated_word_indices, alpha_list, alpha_logits_list, context_projected, activation_maps_flat
    else:
        # doubly stochastic regularization
        # enforces model to pay equal attention to entire image
        if alpha_regularization > 0:
            # convert list to (batch_size, 16, 196)
            alpha_list_ = tf.squeeze(
                tf.transpose(tf.stack(alpha_list), (1, 0, 2, 3)),
                )
            # convert list to (batch_size, 16)
            mask_list_ = tf.to_float(tf.transpose(tf.stack(mask_list), (1, 0)))

            masked_alphas = tf.stack(
                [alpha_list_[:, :, d]*mask_list_ for d in range(alpha_list_.get_shape().as_list()[-1])]
                )

            # convert to (batch_size, 16, 196)
            masked_alphas = tf.transpose(masked_alphas, (1, 2, 0))

            # take sum across sequence dimension
            # leave out first word as it is sentence start
            alphas_all = tf.reduce_sum(masked_alphas[:, 1:, :], 1)

            alpha_reg = alpha_regularization * tf.reduce_sum((1 - alphas_all) ** 2)
            loss += alpha_reg

        # mean batch loss
        loss = loss / tf.reduce_sum(tf.cast(mask_batch, tf.float32))
        tf.summary.scalar('train_loss', loss)

        return loss, generated_word_indices
