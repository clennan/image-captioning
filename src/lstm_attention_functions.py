
import numpy as np
import tensorflow as tf


def lstm_initializations(
    dict_params,
    emb_dim,
    hidden_state_dim,
    num_filters,
    num_words,
    word_bias_init=None,
    ):

    with tf.variable_scope('lstm'):

        with tf.variable_scope('embedding_W'):
            dict_params['embedding_W'], dict_params['embedding_b'] = weights_init(
                shape=[emb_dim, hidden_state_dim*4])
            tf.summary.histogram("weights", dict_params['embedding_W'])
            tf.summary.histogram("bias", dict_params['embedding_b'])

        with tf.variable_scope('hidden_U'):
            dict_params['hidden_U'], _ = weights_init(
                shape=[hidden_state_dim, hidden_state_dim*4],
                orthogonal=True,
                )
            tf.summary.histogram("weights", dict_params['hidden_U'])

        # Wc in paper code
        with tf.variable_scope('weighted_context_W'):
            dict_params['weighted_context_W'], _ = weights_init(
                shape=[num_filters, hidden_state_dim*4],
                )
            tf.summary.histogram("weights", dict_params['weighted_context_W'])

        with tf.variable_scope('memory'):
            dict_params['init_memory_state_W'], dict_params['init_memory_state_b'] = weights_init(
                shape=[num_filters, hidden_state_dim],
                )
            tf.summary.histogram("weights", dict_params['init_memory_state_W'])
            tf.summary.histogram("bias", dict_params['init_memory_state_b'])

        with tf.variable_scope('hidden_state'):
            dict_params['init_hidden_state_W'], dict_params['init_hidden_state_b'] = weights_init(
                shape=[num_filters, hidden_state_dim],
                )
            tf.summary.histogram("weights", dict_params['init_hidden_state_W'])
            tf.summary.histogram("bias", dict_params['init_hidden_state_b'])

        with tf.variable_scope('decode_lstm'):
            dict_params['decode_lstm_W'], dict_params['decode_lstm_b'] = weights_init(
                shape=[hidden_state_dim, emb_dim],
                )
            tf.summary.histogram("weights", dict_params['decode_lstm_W'])
            tf.summary.histogram("bias", dict_params['decode_lstm_b'])

        with tf.variable_scope('decode_word'):
            dict_params['decode_word_W'], _ = weights_init(
                shape=[emb_dim, num_words],
                )
            tf.summary.histogram("weights", dict_params['decode_word_W'])

            if word_bias_init is None:
                dict_params['decode_word_b'] = tf.Variable(
                    tf.zeros([num_words]), name='b',
                    )
            else:
                dict_params['decode_word_b'] = tf.Variable(
                    word_bias_init.astype(np.float32), name='b',
                    )

    return dict_params


def initial_lstm_states(dict_params, mean_context, dropout_lstm_init=0):
    hidden_state_prev = tf.nn.tanh(tf.matmul(
            mean_context, dict_params['init_hidden_state_W'],
            ) + dict_params['init_hidden_state_b'])

    memory_prev = tf.nn.tanh(tf.matmul(
        mean_context, dict_params['init_memory_state_W'],
        ) + dict_params['init_memory_state_b'])

    return hidden_state_prev, memory_prev


def soft_attention_initializations(
    dict_params,
    num_filters,
    hidden_state_dim,
    ):
    '''
    running example: vgg16 conv5_3 [14, 14, 512] --> [196, 512]
    for each 196 'pixels' a_i, the attention mechanism finds weight alpha_i
    each pixel a_i is represented by 512 filters
    '''
    # the attention model is trained for each pixel
    # i.e. same weight matrix [num_filters, num_filters] is applied to each pixel with dim 512
    with tf.variable_scope('soft_attention'):
        # Wc_att in paper code
        with tf.variable_scope('soft_attention_pixel'):
            dict_params['att_pixel_W'], dict_params['att_pixel_b'] = weights_init(
                shape=[num_filters, num_filters])

            tf.summary.histogram("weights", dict_params['att_pixel_W'])
            tf.summary.histogram("bias", dict_params['att_pixel_b'])

        # Wd_att in paper code
        with tf.variable_scope('soft_attention_hidden'):
            dict_params['att_hidden_W'], _ = weights_init(
                shape=[hidden_state_dim, num_filters])

            tf.summary.histogram("weights", dict_params['att_hidden_W'])


        with tf.variable_scope('soft_attention'):
            # U_att in paper code
            dict_params['att_W'], dict_params['att_b'] = weights_init(
                shape=[num_filters, 1])

            tf.summary.histogram("weights", dict_params['att_W'])
            tf.summary.histogram("bias", dict_params['att_b'])

        with tf.variable_scope('beta_gating_scalar'):
            dict_params['att_beta_W'], dict_params['att_beta_b'] = weights_init(
                shape=[hidden_state_dim, 1])

            tf.summary.histogram("weights", dict_params['att_beta_W'])
            tf.summary.histogram("bias", dict_params['att_beta_b'])

    return dict_params


def lstm_step(dict_params,
              word_embedded_prev,
              hidden_state_prev,
              memory_prev,
              dropout_lstm,
              weighted_context,
              ):

    if dropout_lstm > 0:
        word_embedded_prev = tf.nn.dropout(word_embedded_prev, dropout_lstm)

    # compute LSTM affine transformations and apply non-linearities
    hidden_state_proj = tf.matmul(
        hidden_state_prev, dict_params['hidden_U'],
        )  # line 424 in paper code

    word_embedded_proj = tf.matmul(
        word_embedded_prev, dict_params['embedding_W'],
        ) + dict_params['embedding_b']

    lstm_preactive = hidden_state_proj + word_embedded_proj

    # get gating scalar beta
    beta = tf.nn.sigmoid(tf.matmul(
        hidden_state_prev, dict_params['att_beta_W'],
        ) + dict_params['att_beta_b'])

    weighted_context = tf.multiply(
        beta, weighted_context, name='selected_context',
        )

    lstm_preactive += tf.matmul(
        weighted_context, dict_params['weighted_context_W'],
        )

    i, f, o, new_c = tf.split(lstm_preactive, 4, 1)

    i = tf.sigmoid(i)
    f = tf.sigmoid(f)
    o = tf.sigmoid(o)
    new_c = tf.tanh(new_c)

    memory = f * memory_prev + i * new_c  # c_t in the paper
    hidden_state = o * tf.nn.tanh(memory)  # h_t in the paper

    # map from hidden_state_dim to emb_dim and apply dropout for regularization
    logits = tf.matmul(
        hidden_state, dict_params['decode_lstm_W'],
        ) + dict_params['decode_lstm_b']

    logits = tf.nn.relu(logits)

    if dropout_lstm > 0:
        logits = tf.nn.dropout(logits, dropout_lstm)

    # map from embedding space to num_words space
    logits = tf.matmul(
        logits, dict_params['decode_word_W'],
        ) + dict_params['decode_word_b']

    return logits, hidden_state, memory


def soft_attention_step(dict_params,
                        context_projected,
                        hidden_state_prev,
                        activation_maps_flat,
                        num_filters,
                        num_pixels,
                        ):
    '''
    Attention mechanism as per Bahdanau https://arxiv.org/abs/1409.0473
    called soft attention mechanism in Show, Attend, and Tell
    one-layer feed-forward network
    '''
    # project hidden state from previous period to context dimension, i.e. [1, 256] --> [1, 512]
    # line 392 in paper code
    hidden_state_projected = tf.matmul(
        hidden_state_prev, dict_params['att_hidden_W'],
        )

    # add dimension so that hidden state projection can be added
    # to each pixel representation in context_projected
    # i.e. [1, 512] --> [1, 1, 512]
    hidden_state_projected = tf.expand_dims(hidden_state_projected, 1)

    # add hidden state projection to context projected
    # see line 393 https://github.com/kelvinxu/arctic-captions/blob/master/capgen.py
    context_combined = context_projected + hidden_state_projected

    # apply tanh to context, see equation (1) in SAT paper
    # line 396 in paper code
    context_combined = tf.nn.tanh(context_combined)  # [batch_size, 196, 512]

    # flatten to [batch_size * 196, 512]
    context_combined_flat = tf.reshape(context_combined, [-1, num_filters])

    # project for each pixel the context to a scalar, i.e. [batch_size * 196, 512] --> [batch_size * 196, 1]
    alpha_logits = tf.matmul(
        context_combined_flat, dict_params['att_W'],
        ) + dict_params['att_b']

    # reshape back to [batch_size, 196]
    alpha_logits = tf.reshape(alpha_logits, [-1, num_pixels])

    # calculate alpha as per equation (5) in SAT paper
    alpha = tf.nn.softmax(alpha_logits)

    # add dimension to alpha so that context can be weighted_context, [batch_size, 196] --> [batch_size, 196, 1]
    alpha = tf.expand_dims(alpha, 2)

    # calculate weighted context, i.e. apply alpha weight to each pixel
    # [batch_size, 196, 512] * [batch_size, 196, 1] --> [batch_size, 196, 512]
    # i.e. weights get applied to each pixel for all 512 filters
    weighted_context = activation_maps_flat * alpha

    # sum over the pixels as in equation (13) in SAT paper
    # [batch_size, 196, 512] --> [batch_size, 512]
    weighted_context = tf.reduce_sum(weighted_context, 1)

    return weighted_context, alpha, alpha_logits


def weights_init(shape, orthogonal=False, scale=0.01):
    if orthogonal:
        W = tf.get_variable(
            name='U', shape=shape, initializer=tf.orthogonal_initializer(),
            )
        b = tf.Variable(tf.zeros([shape[-1]]), name='b_u')

    else:
        W = tf.get_variable(
            name='W', shape=shape,
            initializer=tf.contrib.layers.xavier_initializer(),
            )
        b = tf.Variable(tf.zeros([shape[-1]]), name='b')

    return (W, b)
