import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from util.ops import shape_list

class Emt_Disc:

  def __init__(self, inputs, is_training, hparams=None, scope='emt_disc'):

    self._hparams = hparams
    filters = [32, 32, 64, 64, 128, 128]
    kernel_size = (3, 3)
    strides = (2,2)

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

      encoder_cell = GRUCell(128)
      ref_outputs = tf.expand_dims(inputs,axis=-1)

      # CNN stack
      for i, channel in enumerate(filters):
        ref_outputs = conv2d(ref_outputs, channel, kernel_size, strides, tf.nn.relu, is_training, 'conv2d_%d' % i)

      shapes = shape_list(ref_outputs)

      ref_outputs = tf.reshape(
        ref_outputs,
        shapes[:-2] + [shapes[2] * shapes[3]])
      # RNN
      encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
        encoder_cell,
        ref_outputs,
        dtype=tf.float32)

      emb = tf.layers.dense(encoder_outputs[:,-1,:], 128, activation=tf.nn.tanh) # [N, 128]
      self.logit = tf.layers.dense(emb, 4)
      self.emb = tf.expand_dims(emb, axis=1) # [N,1,128]

    # return emt_logit, emt_emb

def conv1d(inputs, kernel_size, channels, activation, is_training, scope):
  with tf.variable_scope(scope):
    conv1d_output = tf.layers.conv1d(
      inputs,
      filters=channels,
      kernel_size=kernel_size,
      activation=activation,
      padding='same')
    return tf.layers.batch_normalization(conv1d_output, training=is_training)

def conv2d(inputs, filters, kernel_size, strides, activation, is_training, scope):
  with tf.variable_scope(scope):
    conv2d_output = tf.layers.conv2d(
      inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding='same')
    conv2d_output = tf.layers.batch_normalization(conv2d_output, training=is_training)
    if activation is not None:
      conv2d_output = activation(conv2d_output)
    return conv2d_output









