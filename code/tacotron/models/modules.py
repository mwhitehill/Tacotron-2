# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops


class ReferenceEncoder:
	def __init__(self, filters, kernel_size, strides, is_training, scope, depth, all_outputs=False, emt_ref_gru=None):
		self.filters = filters
		self.kernel_size = kernel_size
		self.strides = strides
		self.is_training = is_training
		self.scope = scope
		self.depth = depth
		self.all_outputs = all_outputs
		self.emt_ref_gru = emt_ref_gru

	def __call__(self, inputs):

		with tf.variable_scope(self.scope):
			ref_outputs = tf.expand_dims(inputs,axis=-1)
			# CNN stack
			for i, channel in enumerate(self.filters):
				strides = self.strides[i] if isinstance(self.strides,list) else self.strides
				ref_outputs = conv2d(ref_outputs, channel, self.kernel_size, strides, tf.nn.relu, self.is_training, 'conv2d_%d' % i)

			shapes = shape_list(ref_outputs)

			ref_outputs = tf.reshape(
				ref_outputs,
				shapes[:-2] + [shapes[2] * shapes[3]])

			if self.all_outputs:
				assert(self.emt_ref_gru in ['gru','gru_multi','none'])
				if self.emt_ref_gru == 'gru':
					with tf.variable_scope('fw_RNN'.format(self.scope)):
						fw_cell = tf.nn.rnn_cell.GRUCell(self.depth)
					with tf.variable_scope('bw_RNN'.format(self.scope)):
						bw_cell = tf.nn.rnn_cell.GRUCell(self.depth)
					ref_outputs, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell, ref_outputs, dtype=tf.float32)
					ref_outputs = tf.concat([ref_outputs[0],ref_outputs[1]],axis=-1)
				elif self.emt_ref_gru == 'gru_multi':
					gru_outputs = None
					for i in range(8):
						with tf.variable_scope('gru_{}'.format(i)):
							encoder_outputs, encoder_state = tf.nn.dynamic_rnn(tf.nn.rnn_cell.GRUCell(self.depth),ref_outputs,dtype=tf.float32)
							reference_state = tf.layers.dense(encoder_outputs[:, -1, :], 128, activation=tf.nn.tanh)  # [N, 128]
							reference_state = tf.expand_dims(reference_state,axis=1)
						gru_outputs = tf.concat([gru_outputs,reference_state],axis=1) if gru_outputs is not None else reference_state
					ref_outputs = gru_outputs

				#else - just use CNN output
				return(ref_outputs)
			else:
			# RNN
				encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
					tf.nn.rnn_cell.GRUCell(self.depth),
					ref_outputs,
					dtype=tf.float32)

				reference_state = tf.layers.dense(encoder_outputs[:,-1,:], 128, activation=tf.nn.tanh) # [N, 128]
				return reference_state

class ReferenceEncoderAdaIn:
	def __init__(self, filters, kernel_size, strides, is_training, scope, depth):
		self.filters = filters
		self.kernel_size = kernel_size
		self.strides = strides
		self.is_training = is_training
		self.scope = scope
		self.depth = depth

	def __call__(self, inputs_spk, inputs_emt):

		with tf.variable_scope(self.scope):
			ref_outputs_spk = tf.expand_dims(inputs_spk, axis=-1)
			ref_outputs_emt = tf.expand_dims(inputs_emt, axis=-1)

			# CNN stack
			for i, channel in enumerate(self.filters):
				strides = self.strides[i] if isinstance(self.strides,list) else self.strides
				ref_outputs_spk = conv2d(ref_outputs_spk, channel, self.kernel_size, strides, tf.nn.relu, self.is_training,
																 'conv2d_%d' % i,batch_norm=False)
				ref_outputs_emt = conv2d(ref_outputs_emt, channel, self.kernel_size, strides, tf.nn.relu, self.is_training,
																 'conv2d_%d' % i, batch_norm=False)

			ref_outputs_spk_pre_norm = ref_outputs_spk
			mean_spk, var_spk = tf.nn.moments(ref_outputs_spk, axes=[1,2],keep_dims=True)
			mean_emt, var_emt = tf.nn.moments(ref_outputs_emt, axes=[1,2],keep_dims=True)
			# ref_outputs_spk = tf.nn.batch_normalization(ref_outputs_spk, mean=mean_spk, variance=var_spk, offset=mean_emt, scale=var_emt,
			# 													variance_epsilon=1e-9)
			ref_outputs_spk_norm = tf.nn.batch_normalization(ref_outputs_spk, mean=mean_spk, variance=var_spk,
																											 offset=mean_emt, scale=var_emt,
																											 variance_epsilon=1e-9)

			ref_outputs_spk = ref_outputs_spk * .9 + ref_outputs_spk_norm * .1

			shapes = shape_list(ref_outputs_spk)
			ref_outputs_spk = tf.reshape(ref_outputs_spk, shapes[:-2] + [shapes[2] * shapes[3]])

			# RNN
			encoder_outputs, encoder_state = tf.nn.dynamic_rnn(tf.nn.rnn_cell.GRUCell(self.depth), ref_outputs_spk, dtype=tf.float32)

			reference_state = tf.layers.dense(encoder_outputs[:,-1,:], 128, activation=tf.nn.tanh) # [N, 128]
			return reference_state, ref_outputs_spk_pre_norm


class HighwayNet:
	def __init__(self, units, name=None):
		self.units = units
		self.scope = 'HighwayNet' if name is None else name

		self.H_layer = tf.layers.Dense(units=self.units, activation=tf.nn.relu, name='H')
		self.T_layer = tf.layers.Dense(units=self.units, activation=tf.nn.sigmoid, name='T', bias_initializer=tf.constant_initializer(-1.))

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
			H = self.H_layer(inputs)
			T = self.T_layer(inputs)
			return H * T + inputs * (1. - T)


class CBHG:
	def __init__(self, K, conv_channels, pool_size, projections, projection_kernel_size, n_highwaynet_layers, highway_units, rnn_units, bnorm, is_training, name=None):
		self.K = K
		self.conv_channels = conv_channels
		self.pool_size = pool_size

		self.projections = projections
		self.projection_kernel_size = projection_kernel_size
		self.bnorm = bnorm

		self.is_training = is_training
		self.scope = 'CBHG' if name is None else name

		self.highway_units = highway_units
		self.highwaynet_layers = [HighwayNet(highway_units, name='{}_highwaynet_{}'.format(self.scope, i+1)) for i in range(n_highwaynet_layers)]
		self._fw_cell = tf.nn.rnn_cell.GRUCell(rnn_units, name='{}_forward_RNN'.format(self.scope))
		self._bw_cell = tf.nn.rnn_cell.GRUCell(rnn_units, name='{}_backward_RNN'.format(self.scope))

	def __call__(self, inputs, input_lengths):
		with tf.variable_scope(self.scope):
			with tf.variable_scope('conv_bank'):
				#Convolution bank: concatenate on the last axis to stack channels from all convolutions
				#The convolution bank uses multiple different kernel sizes to have many insights of the input sequence
				#This makes one of the strengths of the CBHG block on sequences.
				conv_outputs = tf.concat(
					[conv1d(inputs, k, self.conv_channels, tf.nn.relu, self.is_training, 0., self.bnorm, 'conv1d_{}'.format(k)) for k in range(1, self.K+1)],
					axis=-1
					)

			#Maxpooling (dimension reduction, Using max instead of average helps finding "Edges" in mels)
			maxpool_output = tf.layers.max_pooling1d(
				conv_outputs,
				pool_size=self.pool_size,
				strides=1,
				padding='same')

			#Two projection layers
			proj1_output = conv1d(maxpool_output, self.projection_kernel_size, self.projections[0], tf.nn.relu, self.is_training, 0., self.bnorm, 'proj1')
			proj2_output = conv1d(proj1_output, self.projection_kernel_size, self.projections[1], lambda _: _, self.is_training, 0., self.bnorm, 'proj2')

			#Residual connection
			highway_input = proj2_output + inputs

			#Additional projection in case of dimension mismatch (for HighwayNet "residual" connection)
			if highway_input.shape[2] != self.highway_units:
				highway_input = tf.layers.dense(highway_input, self.highway_units)

			#4-layer HighwayNet
			for highwaynet in self.highwaynet_layers:
				highway_input = highwaynet(highway_input)
			rnn_input = highway_input

			#Bidirectional RNN
			outputs, states = tf.nn.bidirectional_dynamic_rnn(
				self._fw_cell,
				self._bw_cell,
				rnn_input,
				sequence_length=input_lengths,
				dtype=tf.float32)
			return tf.concat(outputs, axis=2) #Concat forward and backward outputs


class ZoneoutLSTMCell(tf.nn.rnn_cell.RNNCell):
	'''Wrapper for tf LSTM to create Zoneout LSTM Cell

	inspired by:
	https://github.com/teganmaharaj/zoneout/blob/master/zoneout_tensorflow.py

	Published by one of 'https://arxiv.org/pdf/1606.01305.pdf' paper writers.

	Many thanks to @Ondal90 for pointing this out. You sir are a hero!
	'''
	def __init__(self, num_units, is_training, zoneout_factor_cell=0., zoneout_factor_output=0., state_is_tuple=True, name=None):
		'''Initializer with possibility to set different zoneout values for cell/hidden states.
		'''
		zm = min(zoneout_factor_output, zoneout_factor_cell)
		zs = max(zoneout_factor_output, zoneout_factor_cell)

		if zm < 0. or zs > 1.:
			raise ValueError('One/both provided Zoneout factors are not in [0, 1]')

		self._cell = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=state_is_tuple)#, name=name)
		self._zoneout_cell = zoneout_factor_cell
		self._zoneout_outputs = zoneout_factor_output
		self.is_training = is_training
		self.state_is_tuple = state_is_tuple

	@property
	def state_size(self):
		return self._cell.state_size

	@property
	def output_size(self):
		return self._cell.output_size

	def __call__(self, inputs, state, scope=None):
		'''Runs vanilla LSTM Cell and applies zoneout.
		'''
		#Apply vanilla LSTM
		output, new_state = self._cell(inputs, state, scope)

		if self.state_is_tuple:
			(prev_c, prev_h) = state
			(new_c, new_h) = new_state
		else:
			num_proj = self._cell._num_units if self._cell._num_proj is None else self._cell._num_proj
			prev_c = tf.slice(state, [0, 0], [-1, self._cell._num_units])
			prev_h = tf.slice(state, [0, self._cell._num_units], [-1, num_proj])
			new_c = tf.slice(new_state, [0, 0], [-1, self._cell._num_units])
			new_h = tf.slice(new_state, [0, self._cell._num_units], [-1, num_proj])

		#Apply zoneout
		if self.is_training:
			#nn.dropout takes keep_prob (probability to keep activations) not drop_prob (probability to mask activations)!
			c = (1 - self._zoneout_cell) * tf.nn.dropout(new_c - prev_c, (1 - self._zoneout_cell)) + prev_c
			h = (1 - self._zoneout_outputs) * tf.nn.dropout(new_h - prev_h, (1 - self._zoneout_outputs)) + prev_h

		else:
			c = (1 - self._zoneout_cell) * new_c + self._zoneout_cell * prev_c
			h = (1 - self._zoneout_outputs) * new_h + self._zoneout_outputs * prev_h

		new_state = tf.nn.rnn_cell.LSTMStateTuple(c, h) if self.state_is_tuple else tf.concat(1, [c, h])

		return output, new_state


class EncoderConvolutions:
	"""Encoder convolutional layers used to find local dependencies in inputs characters.
	"""
	def __init__(self, is_training, hparams, activation=tf.nn.relu, scope=None):
		"""
		Args:
			is_training: Boolean, determines if the model is training or in inference to control dropout
			kernel_size: tuple or integer, The size of convolution kernels
			channels: integer, number of convolutional kernels
			activation: callable, postnet activation function for each convolutional layer
			scope: Postnet scope.
		"""
		super(EncoderConvolutions, self).__init__()
		self.is_training = is_training

		self.kernel_size = hparams.enc_conv_kernel_size
		self.channels = hparams.enc_conv_channels
		self.activation = activation
		self.scope = 'enc_conv_layers' if scope is None else scope
		self.drop_rate = hparams.tacotron_dropout_rate
		self.enc_conv_num_layers = hparams.enc_conv_num_layers
		self.bnorm = hparams.batch_norm_position

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
			x = inputs
			for i in range(self.enc_conv_num_layers):
				x = conv1d(x, self.kernel_size, self.channels, self.activation,
					self.is_training, self.drop_rate, self.bnorm, 'conv_layer_{}_'.format(i + 1)+self.scope)
		return x


class EncoderRNN:
	"""Encoder bidirectional one layer LSTM
	"""
	def __init__(self, is_training, size=256, zoneout=0.1, scope=None):
		"""
		Args:
			is_training: Boolean, determines if the model is training or in inference to control zoneout
			size: integer, the number of LSTM units for each direction
			zoneout: the zoneout factor
			scope: EncoderRNN scope.
		"""
		super(EncoderRNN, self).__init__()
		self.is_training = is_training

		self.size = size
		self.zoneout = zoneout
		self.scope = 'encoder_LSTM' if scope is None else scope

		#Create forward LSTM Cell
		self._fw_cell = ZoneoutLSTMCell(size, is_training,
			zoneout_factor_cell=zoneout,
			zoneout_factor_output=zoneout,
			name='encoder_fw_LSTM')

		#Create backward LSTM Cell
		self._bw_cell = ZoneoutLSTMCell(size, is_training,
			zoneout_factor_cell=zoneout,
			zoneout_factor_output=zoneout,
			name='encoder_bw_LSTM')

	def __call__(self, inputs, input_lengths):
		with tf.variable_scope(self.scope):
			outputs, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
				self._fw_cell,
				self._bw_cell,
				inputs,
				sequence_length=input_lengths,
				dtype=tf.float32,
				swap_memory=True)

			return tf.concat(outputs, axis=2) # Concat and return forward + backward outputs


class Prenet:
	"""Two fully connected layers used as an information bottleneck for the attention.
	"""
	def __init__(self, is_training, layers_sizes=[256, 256], drop_rate=0.5, activation=tf.nn.relu, scope=None):
		"""
		Args:
			layers_sizes: list of integers, the length of the list represents the number of pre-net
				layers and the list values represent the layers number of units
			activation: callable, activation functions of the prenet layers.
			scope: Prenet scope.
		"""
		super(Prenet, self).__init__()
		self.drop_rate = drop_rate

		self.layers_sizes = layers_sizes
		self.activation = activation
		self.is_training = is_training

		self.scope = 'prenet' if scope is None else scope

	def __call__(self, inputs):
		x = inputs

		with tf.variable_scope(self.scope):
			for i, size in enumerate(self.layers_sizes):
				dense = tf.layers.dense(x, units=size, activation=self.activation,
					name='dense_{}'.format(i + 1))
				#The paper discussed introducing diversity in generation at inference time
				#by using a dropout of 0.5 only in prenet layers (in both training and inference).
				x = tf.layers.dropout(dense, rate=self.drop_rate, training=True,
					name='dropout_{}'.format(i + 1) + self.scope)
		return x


class DecoderRNN:
	"""Decoder two uni directional LSTM Cells
	"""
	def __init__(self, is_training, layers=2, size=1024, zoneout=0.1, scope=None):
		"""
		Args:
			is_training: Boolean, determines if the model is in training or inference to control zoneout
			layers: integer, the number of LSTM layers in the decoder
			size: integer, the number of LSTM units in each layer
			zoneout: the zoneout factor
		"""
		super(DecoderRNN, self).__init__()
		self.is_training = is_training

		self.layers = layers
		self.size = size
		self.zoneout = zoneout
		self.scope = 'decoder_rnn' if scope is None else scope

		#Create a set of LSTM layers
		self.rnn_layers = [ZoneoutLSTMCell(size, is_training,
			zoneout_factor_cell=zoneout,
			zoneout_factor_output=zoneout,
			name='decoder_LSTM_{}'.format(i+1)) for i in range(layers)]

		self._cell = tf.contrib.rnn.MultiRNNCell(self.rnn_layers, state_is_tuple=True)

	def __call__(self, inputs, states):
		with tf.variable_scope(self.scope):
			return self._cell(inputs, states)


class FrameProjection:
	"""Projection layer to r * num_mels dimensions or num_mels dimensions
	"""
	def __init__(self, shape=80, activation=None, scope=None):
		"""
		Args:
			shape: integer, dimensionality of output space (r*n_mels for decoder or n_mels for postnet)
			activation: callable, activation function
			scope: FrameProjection scope.
		"""
		super(FrameProjection, self).__init__()

		self.shape = shape
		self.activation = activation

		self.scope = 'Linear_projection' if scope is None else scope
		self.dense = tf.layers.Dense(units=shape, activation=activation, name='projection_{}'.format(self.scope))

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
			#If activation==None, this returns a simple Linear projection
			#else the projection will be passed through an activation function
			# output = tf.layers.dense(inputs, units=self.shape, activation=self.activation,
			# 	name='projection_{}'.format(self.scope))
			output = self.dense(inputs)

			return output


class StopProjection:
	"""Projection to a scalar and through a sigmoid activation
	"""
	def __init__(self, is_training, shape=1, activation=tf.nn.sigmoid, scope=None):
		"""
		Args:
			is_training: Boolean, to control the use of sigmoid function as it is useless to use it
				during training since it is integrate inside the sigmoid_crossentropy loss
			shape: integer, dimensionality of output space. Defaults to 1 (scalar)
			activation: callable, activation function. only used during inference
			scope: StopProjection scope.
		"""
		super(StopProjection, self).__init__()
		self.is_training = is_training

		self.shape = shape
		self.activation = activation
		self.scope = 'stop_token_projection' if scope is None else scope

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
			output = tf.layers.dense(inputs, units=self.shape,
				activation=None, name='projection_{}'.format(self.scope))

			#During training, don't use activation as it is integrated inside the sigmoid_cross_entropy loss function
			if self.is_training:
				return output
			return self.activation(output)


class Postnet:
	"""Postnet that takes final decoder output and fine tunes it (using vision on past and future frames)
	"""
	def __init__(self, is_training, hparams, activation=tf.nn.tanh, scope=None):
		"""
		Args:
			is_training: Boolean, determines if the model is training or in inference to control dropout
			kernel_size: tuple or integer, The size of convolution kernels
			channels: integer, number of convolutional kernels
			activation: callable, postnet activation function for each convolutional layer
			scope: Postnet scope.
		"""
		super(Postnet, self).__init__()
		self.is_training = is_training

		self.kernel_size = hparams.postnet_kernel_size
		self.channels = hparams.postnet_channels
		self.activation = activation
		self.scope = 'postnet_convolutions' if scope is None else scope
		self.postnet_num_layers = hparams.postnet_num_layers
		self.drop_rate = hparams.tacotron_dropout_rate
		self.bnorm = hparams.batch_norm_position

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
			x = inputs
			for i in range(self.postnet_num_layers - 1):
				x = conv1d(x, self.kernel_size, self.channels, self.activation,
					self.is_training, self.drop_rate, self.bnorm, 'conv_layer_{}_'.format(i + 1)+self.scope)
			x = conv1d(x, self.kernel_size, self.channels, lambda _: _, self.is_training, self.drop_rate, self.bnorm,
				'conv_layer_{}_'.format(5)+self.scope)
		return x


def conv1d(inputs, kernel_size, channels, activation, is_training, drop_rate, bnorm, scope):
	assert bnorm in ('before', 'after')
	with tf.variable_scope(scope):
		conv1d_output = tf.layers.conv1d(
			inputs,
			filters=channels,
			kernel_size=kernel_size,
			activation=activation if bnorm == 'after' else None,
			padding='same')
		batched = tf.layers.batch_normalization(conv1d_output, training=is_training)
		activated = activation(batched) if bnorm == 'before' else batched
		return tf.layers.dropout(activated, rate=drop_rate, training=is_training,
								name='dropout_{}'.format(scope))

def conv2d(inputs, filters, kernel_size, strides, activation, is_training, scope, batch_norm=True):
  with tf.variable_scope(scope):
    conv2d_output = tf.layers.conv2d(
      inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding='same')
    if batch_norm:
      conv2d_output = tf.layers.batch_normalization(conv2d_output, training=is_training)
    if activation is not None:
      conv2d_output = activation(conv2d_output)
    return conv2d_output

def _round_up_tf(x, multiple):
	# Tf version of remainder = x % multiple
	remainder = tf.mod(x, multiple)
	# Tf version of return x if remainder == 0 else x + multiple - remainder
	x_round =  tf.cond(tf.equal(remainder, tf.zeros(tf.shape(remainder), dtype=tf.int32)),
		lambda: x,
		lambda: x + multiple - remainder)

	return x_round

def sequence_mask(lengths, r, expand=True):
	'''Returns a 2-D or 3-D tensorflow sequence mask depending on the argument 'expand'
	'''
	max_len = tf.reduce_max(lengths)
	max_len = _round_up_tf(max_len, tf.convert_to_tensor(r))
	if expand:
		return tf.expand_dims(tf.sequence_mask(lengths, maxlen=max_len, dtype=tf.float32), axis=-1)
	return tf.sequence_mask(lengths, maxlen=max_len, dtype=tf.float32)

def MaskedMSE(targets, outputs, targets_lengths, hparams, mask=None):
	'''Computes a masked Mean Squared Error
	'''

	#[batch_size, time_dimension, 1]
	#example:
	#sequence_mask([1, 3, 2], 5) = [[[1., 0., 0., 0., 0.]],
	#							    [[1., 1., 1., 0., 0.]],
	#							    [[1., 1., 0., 0., 0.]]]
	#Note the maxlen argument that ensures mask shape is compatible with r>1
	#This will by default mask the extra paddings caused by r>1
	if mask is None:
		mask = sequence_mask(targets_lengths, hparams.outputs_per_step, True)

	#[batch_size, time_dimension, channel_dimension(mels)]
	ones = tf.ones(shape=[tf.shape(mask)[0], tf.shape(mask)[1], tf.shape(targets)[-1]], dtype=tf.float32)
	mask_ = mask * ones

	with tf.control_dependencies([tf.assert_equal(tf.shape(targets), tf.shape(mask_))]):
		return tf.losses.mean_squared_error(labels=targets, predictions=outputs, weights=mask_)

def MaskedSigmoidCrossEntropy(targets, outputs, targets_lengths, hparams, mask=None):
	'''Computes a masked SigmoidCrossEntropy with logits
	'''

	#[batch_size, time_dimension]
	#example:
	#sequence_mask([1, 3, 2], 5) = [[1., 0., 0., 0., 0.],
	#							    [1., 1., 1., 0., 0.],
	#							    [1., 1., 0., 0., 0.]]
	#Note the maxlen argument that ensures mask shape is compatible with r>1
	#This will by default mask the extra paddings caused by r>1
	if mask is None:
		mask = sequence_mask(targets_lengths, hparams.outputs_per_step, False)

	with tf.control_dependencies([tf.assert_equal(tf.shape(targets), tf.shape(mask))]):
		#Use a weighted sigmoid cross entropy to measure the <stop_token> loss. Set hparams.cross_entropy_pos_weight to 1
		#will have the same effect as  vanilla tf.nn.sigmoid_cross_entropy_with_logits.
		losses = tf.nn.weighted_cross_entropy_with_logits(targets=targets, logits=outputs, pos_weight=hparams.cross_entropy_pos_weight)

	with tf.control_dependencies([tf.assert_equal(tf.shape(mask), tf.shape(losses))]):
		masked_loss = losses * mask

	return tf.reduce_sum(masked_loss) / tf.count_nonzero(masked_loss, dtype=tf.float32)

def MaskedLinearLoss(targets, outputs, targets_lengths, hparams, mask=None):
	'''Computes a masked MAE loss with priority to low frequencies
	'''

	#[batch_size, time_dimension, 1]
	#example:
	#sequence_mask([1, 3, 2], 5) = [[[1., 0., 0., 0., 0.]],
	#							    [[1., 1., 1., 0., 0.]],
	#							    [[1., 1., 0., 0., 0.]]]
	#Note the maxlen argument that ensures mask shape is compatible with r>1
	#This will by default mask the extra paddings caused by r>1
	if mask is None:
		mask = sequence_mask(targets_lengths, hparams.outputs_per_step, True)

	#[batch_size, time_dimension, channel_dimension(freq)]
	ones = tf.ones(shape=[tf.shape(mask)[0], tf.shape(mask)[1], tf.shape(targets)[-1]], dtype=tf.float32)
	mask_ = mask * ones

	l1 = tf.abs(targets - outputs)
	n_priority_freq = int(2000 / (hparams.sample_rate * 0.5) * hparams.num_freq)

	with tf.control_dependencies([tf.assert_equal(tf.shape(targets), tf.shape(mask_))]):
		masked_l1 = l1 * mask_
		masked_l1_low = masked_l1[:,:,0:n_priority_freq]

	mean_l1 = tf.reduce_sum(masked_l1) / tf.reduce_sum(mask_)
	mean_l1_low = tf.reduce_sum(masked_l1_low) / tf.reduce_sum(mask_)

	return 0.5 * mean_l1 + 0.5 * mean_l1_low

def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i in range(len(static)):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret

class Style_Emb_Disc:
	"""Simple classifier for style embeddings
	"""
	def __init__(self, output_classes, scope):
		"""
		Args:
			layers_sizes: list of integers, the length of the list represents the number of pre-net
				layers and the list values represent the layers number of units
			activation: callable, activation functions of the prenet layers.
			scope: Prenet scope.
		"""
		super(Style_Emb_Disc, self).__init__()
		self.output_classes = output_classes
		self.scope = scope

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
				logit = tf.layers.dense(inputs, self.output_classes)
		return logit

class Style_Emb_Disc_GRU:
	"""Simple classifier for style embeddings
	"""
	def __init__(self, output_classes, scope, depth=128):
		"""
		Args:
			layers_sizes: list of integers, the length of the list represents the number of pre-net
				layers and the list values represent the layers number of units
			activation: callable, activation functions of the prenet layers.
			scope: Prenet scope.
		"""
		super(Style_Emb_Disc_GRU, self).__init__()
		self.output_classes = output_classes
		self.scope = scope
		self.depth = depth

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
				encoder_outputs, encoder_state = tf.nn.dynamic_rnn(tf.nn.rnn_cell.GRUCell(self.depth),inputs,dtype=tf.float32)
				logit = tf.layers.dense(encoder_outputs[:,-1,:], self.output_classes)
		return logit

class FlipGradientBuilder(object):
	def __init__(self):
		self.num_calls = 0

	def __call__(self, x, l=1.0):
		grad_name = "FlipGradient%d" % self.num_calls

		@ops.RegisterGradient(grad_name)
		def _flip_gradients(op, grad):
			return [tf.negative(grad) * l]

		g = tf.get_default_graph()
		with g.gradient_override_map({"Identity": grad_name}):
			y = tf.identity(x)

		self.num_calls += 1
		return y

def focal_loss(y_true, y_pred):
		"""Focal loss for multi-classification
		FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
		Notice: y_pred is probability after softmax
		gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
		d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
		Focal Loss for Dense Object Detection
		https://arxiv.org/abs/1708.02002

		Arguments:
				y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
				y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

		Keyword Arguments:
				gamma {float} -- (default: {2.0})
				alpha {float} -- (default: {4.0})

		Returns:
				[tensor] -- loss.
		"""
		gamma = 2.
		alpha = 4.
		epsilon = 1.e-9

		# y_true = tf.convert_to_tensor(y_true, tf.float32)
		# y_pred = tf.convert_to_tensor(y_pred, tf.float32)

		model_out = tf.add(y_pred, epsilon)
		ce = tf.multiply(y_true, -tf.log(model_out))
		weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
		fl = tf.multiply(alpha, tf.multiply(weight, ce))
		reduced_fl = tf.reduce_max(fl, axis=1)
		return tf.reduce_mean(reduced_fl)