import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from util.ops import shape_list


def prenet(inputs, is_training, layer_sizes=[256, 128], scope=None):
    x = inputs
    drop_rate = 0.5 if is_training else 0.0
    with tf.variable_scope(scope or 'prenet'):
        for i, size in enumerate(layer_sizes):
            dense = tf.layers.dense(x, units=size, activation=tf.nn.relu, name='dense_%d' % (i + 1))
            x = tf.layers.dropout(dense, rate=drop_rate, name='dropout_%d' % (i + 1))
    return x


def encoder_cbhg(inputs, input_lengths, is_training):
    return cbhg(
        inputs,
        input_lengths,
        is_training,
        scope='encoder_cbhg',
        K=16,
        projections=[128, 128])


def post_cbhg(inputs, input_dim, is_training):
    return cbhg(
        inputs,
        None,
        is_training,
        scope='post_cbhg',
        K=8,
        projections=[256, input_dim])


def cbhg(inputs, input_lengths, is_training, scope, K, projections):
    with tf.variable_scope(scope):
        with tf.variable_scope('conv_bank'):
            # Convolution bank: concatenate on the last axis to stack channels from all convolutions
            conv_outputs = tf.concat(
                [conv1d(inputs, k, 128, tf.nn.relu, is_training, 'conv1d_%d' % k) for k in range(1, K + 1)],
                axis=-1
            )

        # Maxpooling:
        maxpool_output = tf.layers.max_pooling1d(
            conv_outputs,
            pool_size=2,
            strides=1,
            padding='same')

        # Two projection layers:
        proj1_output = conv1d(maxpool_output, 3, projections[0], tf.nn.relu, is_training, 'proj_1')
        proj2_output = conv1d(proj1_output, 3, projections[1], None, is_training, 'proj_2')

        # Residual connection:
        highway_input = proj2_output + inputs

        # Handle dimensionality mismatch:
        if highway_input.shape[2] != 128:
            highway_input = tf.layers.dense(highway_input, 128)

        # 4-layer HighwayNet:
        for i in range(4):
            highway_input = highwaynet(highway_input, 'highway_%d' % (i + 1))
        rnn_input = highway_input

        # Bidirectional RNN
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            GRUCell(128),
            GRUCell(128),
            rnn_input,
            sequence_length=input_lengths,
            dtype=tf.float32)
        return tf.concat(outputs, axis=2)  # Concat forward and backward


def highwaynet(inputs, scope):
    with tf.variable_scope(scope):
        H = tf.layers.dense(
            inputs,
            units=128,
            activation=tf.nn.relu,
            name='H')
        T = tf.layers.dense(
            inputs,
            units=128,
            activation=tf.nn.sigmoid,
            name='T',
            bias_initializer=tf.constant_initializer(-1.0))
        return H * T + inputs * (1.0 - T)


def conv1d(inputs, kernel_size, channels, activation, is_training, scope):
    with tf.variable_scope(scope):
        conv1d_output = tf.layers.conv1d(
            inputs,
            filters=channels,
            kernel_size=kernel_size,
            activation=activation,
            padding='same')
        return tf.layers.batch_normalization(conv1d_output, training=is_training)


def conv2d(inputs, kernel_size, channels, strides, activation, is_training, scope):
    with tf.variable_scope(scope):
        conv2d_output = tf.layers.conv2d(
            inputs,
            filters=channels,
            strides=strides,
            kernel_size=kernel_size,
            padding='same')
        conv2d_output = tf.layers.batch_normalization(conv2d_output, training=is_training)
        if activation is not None:
            conv2d_output = activation(conv2d_output)
        return conv2d_output


def reference_encoder(inputs, filters, kernel_size, strides, is_training, scope="reference_encoder"):
    """
    Use 6 x 2-D convolution layers and A single GRU
    """
    with tf.variable_scope(scope):
        # inputs: N x T x M x 1, output N x T x M x C
        outputs = tf.expand_dims(inputs, axis=-1)  # for 2-D conv
        for i, channels in enumerate(filters):
            outputs = conv2d(outputs, kernel_size, channels, strides, tf.nn.relu, is_training, 're_conv2d_%d' % i)

        # reshape to 3 dimension and preserving time resolution

        shapes = shape_list(outputs)
        outputs = tf.reshape(outputs, [shapes[0], shapes[1], shapes[2]*shapes[3]])

        # apply a single rnn layer
        outputs, states = tf.nn.dynamic_rnn(GRUCell(128), outputs, dtype=tf.float32)
        # the last state serves as the reference encoder embedding
        return tf.nn.tanh(outputs), states
