import tensorflow as tf 
from tacotron.utils.symbols import symbols
from infolog import log
from tacotron.models.helpers import TacoTrainingHelper, TacoTestHelper
from tacotron.models.modules import *
from tensorflow.contrib.seq2seq import dynamic_decode
from tacotron.models.Architecture_wrappers import TacotronEncoderCell, TacotronDecoderCell
from tacotron.models.custom_decoder import CustomDecoder
from tacotron.models.attention import LocationSensitiveAttention
from tacotron.models.multihead_attention import MultiheadAttention

import numpy as np

def split_func(x, split_pos):
	rst = []
	start = 0
	# x will be a numpy array with the contents of the placeholder below
	for i in range(split_pos.shape[0]):
		rst.append(x[:,start:start+split_pos[i]])
		start += split_pos[i]
	return rst

class Tacotron():
	"""Tacotron-2 Feature prediction Model.
	"""
	def __init__(self, hparams):
		self._hparams = hparams

	def initialize(self, args, inputs, input_lengths, mel_targets=None, stop_token_targets=None, linear_targets=None,
								 targets_lengths=None, gta=False,global_step=None, is_training=False, is_evaluating=False,
								 split_infos=None, emt_labels=None, spk_labels=None, emt_up_labels=None, spk_up_labels=None, spk_emb=None,
								 ref_mel_emt = None, ref_mel_spk = None, ref_mel_up_emt = None, ref_mel_up_spk = None, use_emt_disc = False,
								 use_spk_disc = False, use_intercross=False, use_unpaired = False, n_emt=None, n_spk=None):
		"""
		Initializes the model for inference
		sets "mel_outputs" and "alignments" fields.
		Args:
			- inputs: int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
			  steps in the input time series, and values are character IDs
			- input_lengths: int32 Tensor with shape [N] where N is batch size and values are the lengths
			of each sequence in inputs.
			- mel_targets: float32 Tensor with shape [N, T_out, M] where N is batch size, T_out is number
			of steps in the output time series, M is num_mels, and values are entries in the mel
			spectrogram. Only needed for training.
		"""
		if mel_targets is None and stop_token_targets is not None:
			raise ValueError('no multi targets were provided but token_targets were given')
		if mel_targets is not None and stop_token_targets is None and not gta:
			raise ValueError('Mel targets are provided without corresponding token_targets')
		if not gta and self._hparams.predict_linear==True and linear_targets is None and is_training:
			raise ValueError('Model is set to use post processing to predict linear spectrograms in training but no linear targets given!')
		if gta and linear_targets is not None:
			raise ValueError('Linear spectrogram prediction is not supported in GTA mode!')
		if is_training and self._hparams.mask_decoder and targets_lengths is None:
			raise RuntimeError('Model set to mask paddings but no targets lengths provided for the mask!')
		if is_training and is_evaluating:
			raise RuntimeError('Model can not be in training and evaluation modes at the same time!')
		if self._hparams.tacotron_use_style_emb_disc and (n_emt==None or n_spk==None):
			raise ValueError('must specify number of emotions and number of speakers!')
		if use_unpaired and not(self._hparams.tacotron_use_style_emb_disc):
			raise ValueError('trying to use unpaired ')
		if args.nat_gan and not(args.unpaired):
			print("USING NATURALNESS GAN WITHOUT UNPAIRED SAMPLES")
		if ref_mel_emt is None and ref_mel_spk is None:
			raise ValueError("must provide references")


		self.use_emt_disc = use_emt_disc
		self.use_spk_disc = use_spk_disc
		self.use_intercross = use_intercross
		self.use_unpaired = use_unpaired
		self.args = args
		self.n_emt = n_emt
		#n_spk=252
		self.n_spk = n_spk

		split_device = '/cpu:0' if self._hparams.tacotron_num_gpus > 1 or self._hparams.split_on_cpu else '/gpu:0'
		with tf.device(split_device):
			hp = self._hparams
			lout_int = [tf.int32]*hp.tacotron_num_gpus
			lout_float = [tf.float32]*hp.tacotron_num_gpus

			tower_input_lengths = tf.split(input_lengths, num_or_size_splits=hp.tacotron_num_gpus, axis=0)
			tower_targets_lengths = tf.split(targets_lengths, num_or_size_splits=hp.tacotron_num_gpus, axis=0) if targets_lengths is not None else targets_lengths
			tower_emt_labels = tf.to_float(tf.split(emt_labels, num_or_size_splits=hp.tacotron_num_gpus, axis=0)) if emt_labels is not None else emt_labels
			tower_spk_labels = tf.to_float(tf.split(spk_labels, num_or_size_splits=hp.tacotron_num_gpus, axis=0))if spk_labels is not None else spk_labels
			tower_emt_up_labels = tf.to_float(tf.split(emt_up_labels, num_or_size_splits=hp.tacotron_num_gpus, axis=0)) if emt_up_labels is not None else emt_up_labels
			tower_spk_up_labels = tf.to_float(tf.split(spk_up_labels, num_or_size_splits=hp.tacotron_num_gpus, axis=0)) if spk_up_labels is not None else spk_up_labels

			p_inputs = tf.py_func(split_func, [inputs, split_infos[:, 0]], lout_int)
			p_spk_emb = tf.py_func(split_func, [spk_emb, split_infos[:, 4]], lout_float) if spk_emb is not None else spk_emb
			p_mel_targets = tf.py_func(split_func, [mel_targets, split_infos[:,1]], lout_float) if mel_targets is not None else mel_targets
			p_stop_token_targets = tf.py_func(split_func, [stop_token_targets, split_infos[:,2]], lout_float) if stop_token_targets is not None else stop_token_targets
			p_linear_targets = tf.py_func(split_func, [linear_targets, split_infos[:,3]], lout_float) if linear_targets is not None else linear_targets

			p_ref_mel_emt = tf.py_func(split_func, [ref_mel_emt, split_infos[:,5]], lout_float) if ref_mel_emt is not None else ref_mel_emt
			p_ref_mel_spk = tf.py_func(split_func, [ref_mel_spk, split_infos[:, 6]],lout_float) if ref_mel_spk is not None else ref_mel_spk
			p_ref_mel_up_emt = tf.py_func(split_func, [ref_mel_up_emt, split_infos[:, 7]],lout_float) if ref_mel_up_emt is not None else ref_mel_up_emt
			p_ref_mel_up_spk = tf.py_func(split_func, [ref_mel_up_spk, split_infos[:, 8]],lout_float) if ref_mel_up_spk is not None else ref_mel_up_spk

			tower_inputs = []
			tower_mel_targets = []
			tower_stop_token_targets = []
			tower_linear_targets = []
			tower_spk_emb = []
			tower_ref_mel_emt = []
			tower_ref_mel_spk = []
			tower_ref_mel_up_emt = []
			tower_ref_mel_up_spk = []

			self.batch_size = tf.shape(inputs)[0]
			self.batch_size_int = int(self._hparams.tacotron_batch_size / self._hparams.tacotron_num_gpus)
			mel_channels = hp.num_mels
			linear_channels = hp.num_freq
			for i in range (hp.tacotron_num_gpus):
				tower_inputs.append(tf.reshape(p_inputs[i], [self.batch_size, -1]))
				if p_spk_emb is not None:
					tower_spk_emb.append(tf.reshape(p_spk_emb[i], [-1, self._hparams.tacotron_spk_emb_dim]))
				if p_mel_targets is not None:
					tower_mel_targets.append(tf.reshape(p_mel_targets[i], [self.batch_size, -1, mel_channels]))
				if p_stop_token_targets is not None:
					tower_stop_token_targets.append(tf.reshape(p_stop_token_targets[i], [self.batch_size, -1]))
				if p_linear_targets is not None:
					tower_linear_targets.append(tf.reshape(p_linear_targets[i], [self.batch_size, -1, linear_channels]))
				if p_ref_mel_emt is not None:
					tower_ref_mel_emt.append(tf.reshape(p_ref_mel_emt[i], [self.batch_size, -1, mel_channels]))
				if p_ref_mel_spk is not None:
					tower_ref_mel_spk.append(tf.reshape(p_ref_mel_spk[i], [self.batch_size, -1, mel_channels]))
				if p_ref_mel_up_emt is not None:
					tower_ref_mel_up_emt.append(tf.reshape(p_ref_mel_up_emt[i], [self.batch_size, -1, mel_channels]))
				if p_ref_mel_up_spk is not None:
					tower_ref_mel_up_spk.append(tf.reshape(p_ref_mel_up_spk[i], [self.batch_size, -1, mel_channels]))

		T2_output_range = (-hp.max_abs_value, hp.max_abs_value) if hp.symmetric_mels else (0, hp.max_abs_value)

		self.tower_decoder_output = []
		self.tower_decoder_output_up = []
		self.tower_alignments = []
		self.tower_stop_token_prediction = []
		self.tower_mel_outputs = []
		self.tower_linear_outputs = []
		self.tower_refnet_out_emt = []
		self.tower_refnet_out_spk = []
		self.tower_style_embeddings = []
		self.tower_style_emb_logit_emt = []
		self.tower_style_emb_logit_spk = []
		self.tower_nat_gan_logits_targets = []
		self.tower_nat_gan_logits_mel_p = []
		if use_unpaired:
			self.tower_nat_gan_logits_mel_up = []

		if use_unpaired:
			self.tower_mel_outputs_up = []
			if self._hparams.tacotron_use_style_emb_disc:
				self.tower_style_emb_logit_up_emt = []
				self.tower_style_emb_logit_up_spk = []
				self.tower_style_emb_logit_mel_out_up_emt = []
				self.tower_style_emb_logit_mel_out_up_spk = []

		if use_unpaired:
			self.tower_refnet_out_up_emt = []
			self.tower_refnet_out_up_spk = []
			self.tower_refnet_outputs_mel_out_up_emt = []
			self.tower_refnet_outputs_mel_out_up_spk = []

		self.tower_embedded_inputs = []
		tower_enc_conv_output_shape = []
		self.tower_encoder_outputs = []
		self.tower_encoder_outputs_up = []
		tower_residual = []
		tower_projected_residual = []
		
		# 1. Declare GPU Devices
		gpus = ["/gpu:{}".format(i) for i in range(hp.tacotron_num_gpus)]
		for i in range(hp.tacotron_num_gpus):
			with tf.device(tf.train.replica_device_setter(ps_tasks=1, ps_device="/cpu:0", worker_device=gpus[i])):
				with tf.variable_scope('inference') as scope:
					assert hp.tacotron_teacher_forcing_mode in ('constant', 'scheduled')
					if hp.tacotron_teacher_forcing_mode == 'scheduled' and is_training:
						assert global_step is not None


					input_len = tower_input_lengths[i] # tf.concat([tower_input_lengths[i],tower_input_lengths[i]], axis=0) if self.use_unpaired else tower_input_lengths[i]
					if mel_targets is not None:
						mel_targets = tower_mel_targets[i] #tf.concat([tower_mel_targets[i], tower_mel_targets[i]], axis=0) if self.use_unpaired else tower_mel_targets[i]

					#GTA is only used for predicting mels to train Wavenet vocoder, so we ommit post processing when doing GTA synthesis
					post_condition = hp.predict_linear and not gta

					# Embeddings ==> [batch_size, sequence_length, embedding_dim]
					self.embedding_table = tf.get_variable(
						'inputs_embedding', [len(symbols), hp.embedding_dim], dtype=tf.float32)
					embedded_inputs = tf.nn.embedding_lookup(self.embedding_table, tower_inputs[i])

					if hp.use_gst:
						# Global style tokens (GST)
						gst_tokens_emt = tf.get_variable('style_tokens_emt', [hp.num_gst, hp.style_embed_depth // hp.num_heads],
																				 dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.5))
						gst_tokens_spk = tf.get_variable('style_tokens_spk', [hp.num_gst, hp.style_embed_depth // hp.num_heads],
																						 dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.5))
						self.gst_tokens = [gst_tokens_emt,gst_tokens_spk]

					#Encoder Cell ==> [batch_size, encoder_steps, encoder_lstm_units]
					encoder_cell = TacotronEncoderCell(
						EncoderConvolutions(is_training, hparams=hp, scope='encoder_convolutions'),
						EncoderRNN(is_training, size=hp.encoder_lstm_units, zoneout=hp.tacotron_zoneout_rate, scope='encoder_LSTM'))

					encoder_outputs = encoder_cell(embedded_inputs, tower_input_lengths[i])

					#For shape visualization purpose
					enc_conv_output_shape = encoder_cell.conv_output_shape

					# Reference encoder
					reference_encoder_emt = ReferenceEncoder(filters=hp.reference_filters, kernel_size=(3, 3),
																							 strides=(2, 2), is_training=is_training, scope='refnet_emt',
																									 depth=hp.reference_depth)  # [N, 128])
					reference_encoder_spk = ReferenceEncoder(filters=hp.reference_filters, kernel_size=(3, 3),
																									 strides=(2, 2), is_training=is_training, scope='refnet_spk',
																									 depth=hp.reference_depth)  # [N, 128])

					#Get references
					refnet_outputs_emt = reference_encoder_emt(tower_ref_mel_emt[i]) # [N, 128]
					refnet_outputs_spk = reference_encoder_spk(tower_ref_mel_spk[i]) # [N, 128]
					if use_unpaired:
						refnet_outputs_up_emt = reference_encoder_emt(tower_ref_mel_up_emt[i])  # [N, 128]
						refnet_outputs_up_spk = reference_encoder_spk(tower_ref_mel_up_spk[i])  # [N, 128]

					# Calculate GST
					if hp.use_gst:
						# Style attention modules
						value_emt = tf.tanh(tf.tile(tf.expand_dims(gst_tokens_emt, axis=0), [self.batch_size, 1, 1]))
						value_spk = tf.tanh(tf.tile(tf.expand_dims(gst_tokens_spk, axis=0), [self.batch_size, 1, 1]))
						style_attention_emt = MultiheadAttention(num_heads=hp.num_heads, num_units=hp.style_att_dim, attention_type=hp.style_att_type, scope='emt')
						style_attention_spk = MultiheadAttention(num_heads=hp.num_heads, num_units=hp.style_att_dim, attention_type=hp.style_att_type, scope='spk')

						#calculate the style embeddings
						style_embeddings_emt = style_attention_emt.multi_head_attention(tf.expand_dims(refnet_outputs_emt, axis=1),value_emt) # [N, 1, 128]
						style_embeddings_spk = style_attention_spk.multi_head_attention(tf.expand_dims(refnet_outputs_spk, axis=1),value_spk) # [N, 1, 128]
						if use_unpaired:
							style_embeddings_up_emt = style_attention_emt.multi_head_attention(tf.expand_dims(refnet_outputs_up_emt, axis=1), value_emt)  # [N, 1, 128]
							style_embeddings_up_spk = style_attention_spk.multi_head_attention(tf.expand_dims(refnet_outputs_up_spk, axis=1), value_spk)  # [N, 1, 128]
					else:
						# just use the embeddings themselves to pass into decoder
						style_embeddings_emt = tf.expand_dims(refnet_outputs_emt, axis=1)  # [N, 1, 128]
						style_embeddings_spk = tf.expand_dims(refnet_outputs_spk, axis=1)  # [N, 1, 128]
						if use_unpaired:
							style_embeddings_up_emt = tf.expand_dims(refnet_outputs_up_emt, axis=1)  # [N, 1, 128]
							style_embeddings_up_spk = tf.expand_dims(refnet_outputs_up_spk, axis=1)  # [N, 1, 128]

					# concat emotion and speaker embeddings
					style_embeddings = tf.concat([style_embeddings_emt, style_embeddings_spk], axis=-1)
					if use_unpaired:
						style_embeddings_up = tf.concat([style_embeddings_up_emt, style_embeddings_up_spk], axis=-1)

					# Concatenate the style embeddings to encoder embeddings
					if use_unpaired: #must come first so can concat enocder outputs before they are concatenated
						style_embeddings_up = tf.tile(style_embeddings_up, [1, shape_list(encoder_outputs)[1], 1])  # [N, T_in, 128]
						encoder_outputs_up = tf.concat([encoder_outputs, style_embeddings_up], axis=-1)

					style_embeddings = tf.tile(style_embeddings, [1, shape_list(encoder_outputs)[1], 1])  # [N, T_in, 128]
					encoder_outputs = tf.concat([encoder_outputs, style_embeddings],axis=-1)

					#Decoder Parts
					#Attention Decoder Prenet
					prenet = Prenet(is_training, layers_sizes=hp.prenet_layers, drop_rate=hp.tacotron_dropout_rate, scope='decoder_prenet')
					#Attention Mechanism
					attention_mechanism = LocationSensitiveAttention(hp.attention_dim, encoder_outputs, hparams=hp, is_training=is_training,
						mask_encoder=hp.mask_encoder, memory_sequence_length=tf.reshape(input_len, [-1]), smoothing=hp.smoothing,
						cumulate_weights=hp.cumulative_weights)
					#Decoder LSTM Cells
					decoder_lstm = DecoderRNN(is_training, layers=hp.decoder_layers,
						size=hp.decoder_lstm_units, zoneout=hp.tacotron_zoneout_rate, scope='decoder_LSTM')
					#Frames Projection layer
					frame_projection = FrameProjection(hp.num_mels * hp.outputs_per_step, scope='linear_transform_projection')
					#<stop_token> projection layer
					stop_projection = StopProjection(is_training or is_evaluating, shape=hp.outputs_per_step, scope='stop_token_projection')


					#Decoder Cell ==> [batch_size, decoder_steps, num_mels * r] (after decoding)
					decoder_cell = TacotronDecoderCell(
						prenet,
						attention_mechanism,
						decoder_lstm,
						frame_projection,
						stop_projection)

					#Define the helper for our decoder
					if is_training or is_evaluating or gta:
						self.helper = TacoTrainingHelper(self.batch_size, mel_targets, hp, gta, is_evaluating, global_step)
					else:
						self.helper = TacoTestHelper(self.batch_size, hp)

					#initial decoder state
					decoder_init_state = decoder_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

					#Only use max iterations at synthesis time
					max_iters = hp.max_iters if not (is_training or is_evaluating) else None

					#Decode
					custom_decoder = CustomDecoder(decoder_cell, self.helper, decoder_init_state)

					(frames_prediction, stop_token_prediction, _), final_decoder_state, _ = dynamic_decode(custom_decoder,
						impute_finished=False,
						maximum_iterations=max_iters,
						swap_memory=hp.tacotron_swap_with_cpu,scope='decoder')


					# Reshape outputs to be one output per entry
					#==> [batch_size, non_reduced_decoder_steps (decoder_steps * r), num_mels]
					decoder_output = tf.reshape(frames_prediction, [self.batch_size, -1, hp.num_mels])
					stop_token_prediction = tf.reshape(stop_token_prediction, [self.batch_size, -1])

					if hp.clip_outputs:
							decoder_output = tf.minimum(tf.maximum(decoder_output, T2_output_range[0] - hp.lower_bound_decay), T2_output_range[1])

					#Postnet
					postnet = Postnet(is_training, hparams=hp, scope='postnet_convolutions')

					#Compute residual using post-net ==> [batch_size, decoder_steps * r, postnet_channels]
					residual = postnet(decoder_output)

					#Project residual to same dimension as mel spectrogram
					#==> [batch_size, decoder_steps * r, num_mels]
					residual_projection = FrameProjection(hp.num_mels, scope='postnet_projection')
					projected_residual = residual_projection(residual)


					#Compute the mel spectrogram
					mel_outputs = decoder_output + projected_residual

					if hp.clip_outputs:
							mel_outputs = tf.minimum(tf.maximum(mel_outputs, T2_output_range[0] - hp.lower_bound_decay), T2_output_range[1])

					##DECODER UNPAIRED
					if use_unpaired:
						# Decoder Parts
						# Attention Decoder Prenet
						prenet_up = Prenet(is_training, layers_sizes=hp.prenet_layers, drop_rate=hp.tacotron_dropout_rate,
														scope='decoder_prenet')
						# Attention Mechanism
						attention_mechanism_up = LocationSensitiveAttention(hp.attention_dim, encoder_outputs_up, hparams=hp,
																														 is_training=is_training,
																														 mask_encoder=hp.mask_encoder,
																														 memory_sequence_length=tf.reshape(input_len, [-1]),
																														 smoothing=hp.smoothing,
																														 cumulate_weights=hp.cumulative_weights)
						# Decoder LSTM Cells
						decoder_lstm_up = DecoderRNN(is_training, layers=hp.decoder_layers,
																			size=hp.decoder_lstm_units, zoneout=hp.tacotron_zoneout_rate,
																			scope='decoder_LSTM')
						# Frames Projection layer
						frame_projection_up = FrameProjection(hp.num_mels * hp.outputs_per_step, scope='linear_transform_projection')
						# <stop_token> projection layer
						stop_projection_up = StopProjection(is_training or is_evaluating, shape=hp.outputs_per_step,scope='stop_token_projection')

						# Decoder Cell ==> [batch_size, decoder_steps, num_mels * r] (after decoding)
						decoder_cell_up = TacotronDecoderCell(
							prenet_up,
							attention_mechanism_up,
							decoder_lstm_up,
							frame_projection_up,
							stop_projection_up)

						# Define the helper for our decoder
						self.helper_up = TacoTrainingHelper(self.batch_size, mel_targets, hp, gta, evaluating=True, global_step=global_step)

						# initial decoder state
						decoder_init_state_up = decoder_cell_up.zero_state(batch_size=self.batch_size, dtype=tf.float32)

						# Only use max iterations at synthesis time
						max_iters_up = hp.max_iters

						# Decode
						custom_decoder_up = CustomDecoder(decoder_cell_up, self.helper_up, decoder_init_state_up)

						(frames_prediction_up, stop_token_prediction_up, _), final_decoder_state_up, _ = dynamic_decode(custom_decoder_up,
																																																	 impute_finished=False,
																																																	 maximum_iterations=max_iters_up,
																																																	 swap_memory=hp.tacotron_swap_with_cpu,
																																																						scope='decoder')

						# Reshape outputs to be one output per entry
						# ==> [batch_size, non_reduced_decoder_steps (decoder_steps * r), num_mels]
						decoder_output_up = tf.reshape(frames_prediction_up, [self.batch_size, -1, hp.num_mels])
						stop_token_prediction_up = tf.reshape(stop_token_prediction_up, [self.batch_size, -1])

						if hp.clip_outputs:
							decoder_output_up = tf.minimum(tf.maximum(decoder_output_up, T2_output_range[0] - hp.lower_bound_decay),
																					T2_output_range[1])

						# Postnet
						postnet_up = Postnet(is_training, hparams=hp, scope='postnet_convolutions')

						# Compute residual using post-net ==> [batch_size, decoder_steps * r, postnet_channels]
						residual_up = postnet_up(decoder_output_up)

						# Project residual to same dimension as mel spectrogram
						# ==> [batch_size, decoder_steps * r, num_mels]
						residual_projection_up = FrameProjection(hp.num_mels, scope='postnet_projection')
						projected_residual_up = residual_projection_up(residual_up)

						# Compute the mel spectrogram
						mel_outputs_up = decoder_output_up + projected_residual_up

						if hp.clip_outputs:
							mel_outputs_up = tf.minimum(tf.maximum(mel_outputs_up, T2_output_range[0] - hp.lower_bound_decay),
																			 T2_output_range[1])


					# if post_condition:
					# 	# Add post-processing CBHG. This does a great job at extracting features from mels before projection to Linear specs.
					# 	post_cbhg = CBHG(hp.cbhg_kernels, hp.cbhg_conv_channels, hp.cbhg_pool_size, [hp.cbhg_projection, hp.num_mels],
					# 		hp.cbhg_projection_kernel_size, hp.cbhg_highwaynet_layers,
					# 		hp.cbhg_highway_units, hp.cbhg_rnn_units, hp.batch_norm_position, is_training, name='CBHG_postnet')
					#
					# 	#[batch_size, decoder_steps(mel_frames), cbhg_channels]
					# 	post_outputs = post_cbhg(mel_outputs, None)
					#
					# 	#Linear projection of extracted features to make linear spectrogram
					# 	linear_specs_projection = FrameProjection(hp.num_freq, scope='cbhg_linear_specs_projection')
					#
					# 	#[batch_size, decoder_steps(linear_frames), num_freq]
					# 	linear_outputs = linear_specs_projection(post_outputs)
					#
					# 	if hp.clip_outputs:
					# 		linear_outputs = tf.minimum(tf.maximum(linear_outputs, T2_output_range[0] - hp.lower_bound_decay), T2_output_range[1])


					#Style Embedding Discrciminator
					style_emb_disc_emt = Style_Emb_Disc(n_emt, scope='style_disc_emt')
					style_emb_disc_spk = Style_Emb_Disc(n_spk, scope='style_disc_spk')

					#Get style embedding logits for normal and upaired references
					style_emb_logit_emt = style_emb_disc_emt(refnet_outputs_emt)
					style_emb_logit_spk = style_emb_disc_spk(refnet_outputs_spk)
					if use_unpaired:
						style_emb_logit_up_emt = style_emb_disc_emt(refnet_outputs_up_emt)
						style_emb_logit_up_spk = style_emb_disc_spk(refnet_outputs_up_spk)

						# If using unpaired, re-encode the mel output and get the logits
						refnet_outputs_mel_out_up_emt = reference_encoder_emt(mel_outputs_up)  # [N, 128] #tf.constant(0.0,shape=[32,128])
						refnet_outputs_mel_out_up_spk = reference_encoder_spk(mel_outputs_up)  # [N, 128] #tf.constant(0.0,shape=[32,128])

						style_emb_logit_mel_out_up_emt = style_emb_disc_emt(refnet_outputs_mel_out_up_emt) #tf.constant(0.0,shape=[1,4])
						style_emb_logit_mel_out_up_spk = style_emb_disc_spk(refnet_outputs_mel_out_up_spk) #tf.constant(0.0, shape=[1,110])


					if self.args.nat_gan:
						nat_gan_enc = ReferenceEncoder(filters=hp.reference_filters, kernel_size=(3, 3),
																					strides=(2, 2),is_training=is_training, scope='nat_gan_enc',
																					 depth = hp.reference_depth)  # [N, 128])

						nat_gan_disc = Style_Emb_Disc(3, scope='nat_gan_disc')

						nat_gan_enc_out_targets = nat_gan_enc(tower_mel_targets[i]) # [N, 128]
						nat_gan_enc_out_mel_p = nat_gan_enc(mel_outputs) # [N, 128]

						nat_gan_logits_targets = nat_gan_disc(nat_gan_enc_out_targets)
						nat_gan_logits_mel_p = nat_gan_disc(nat_gan_enc_out_mel_p)

						if args.unpaired:
							nat_gan_enc_out_mel_up = nat_gan_enc_out_mel_p[self.batch_size:, :]
							nat_gan_enc_out_mel_p = nat_gan_enc_out_mel_p[:self.batch_size, :]

							nat_gan_logits_mel_up = nat_gan_logits_mel_p[self.batch_size:, :]
							nat_gan_logits_mel_p = nat_gan_logits_mel_p[:self.batch_size, :]

					#Grab alignments from the final decoder state
					alignments = tf.transpose(final_decoder_state.alignment_history.stack(), [1, 2, 0])

					self.tower_decoder_output.append(decoder_output)
					self.tower_alignments.append(alignments)
					self.tower_refnet_out_emt.append(refnet_outputs_emt)
					self.tower_refnet_out_spk.append(refnet_outputs_spk)
					self.tower_style_embeddings.append(style_embeddings)
					self.tower_stop_token_prediction.append(stop_token_prediction)
					self.tower_mel_outputs.append(mel_outputs)
					self.tower_style_emb_logit_emt.append(style_emb_logit_emt)
					self.tower_style_emb_logit_spk.append(style_emb_logit_spk)
					if self.args.nat_gan:
						self.tower_nat_gan_logits_targets.append(nat_gan_logits_targets)
						self.tower_nat_gan_logits_mel_p.append(nat_gan_logits_mel_p)
						if self.use_unpaired:
							self.tower_nat_gan_logits_mel_up.append(nat_gan_logits_mel_up)

					if use_unpaired and self._hparams.tacotron_use_style_emb_disc:
						self.tower_style_emb_logit_up_emt.append(style_emb_logit_up_emt)
						self.tower_style_emb_logit_up_spk.append(style_emb_logit_up_spk)
						self.tower_style_emb_logit_mel_out_up_emt.append(style_emb_logit_mel_out_up_emt)
						self.tower_style_emb_logit_mel_out_up_spk.append(style_emb_logit_mel_out_up_spk)

					if use_unpaired:
						self.tower_mel_outputs_up.append(mel_outputs_up)
						self.tower_decoder_output_up.append(decoder_output_up)
						self.tower_refnet_out_up_emt.append(refnet_outputs_up_emt)
						self.tower_refnet_out_up_spk.append(refnet_outputs_up_spk)
						self.tower_refnet_outputs_mel_out_up_emt.append(refnet_outputs_mel_out_up_emt)
						self.tower_refnet_outputs_mel_out_up_spk.append(refnet_outputs_mel_out_up_spk)

					self.tower_embedded_inputs.append(embedded_inputs)
					tower_enc_conv_output_shape.append(enc_conv_output_shape)
					self.tower_encoder_outputs.append(encoder_outputs)
					if use_unpaired:
						self.tower_encoder_outputs_up.append(encoder_outputs_up)
					tower_residual.append(residual)
					tower_projected_residual.append(projected_residual)

					if post_condition:
						self.tower_linear_outputs.append(linear_outputs)
			log('initialisation done {}'.format(gpus[i]))


		if is_training:
			self.ratio = self.helper._ratio
		self.tower_inputs = tower_inputs
		self.tower_input_lengths = tower_input_lengths
		self.tower_mel_targets = tower_mel_targets
		self.tower_linear_targets = tower_linear_targets
		self.tower_targets_lengths = tower_targets_lengths
		self.tower_stop_token_targets = tower_stop_token_targets
		self.tower_emt_labels = tower_emt_labels
		self.tower_spk_labels = tower_spk_labels
		self.tower_emt_up_labels = tower_emt_up_labels
		self.tower_spk_up_labels = tower_spk_up_labels
		self.tower_ref_mel_emt = tower_ref_mel_emt
		self.tower_ref_mel_spk = tower_ref_mel_spk
		if args.unpaired:
			self.tower_ref_mel_up_emt = tower_ref_mel_up_emt
			self.tower_ref_mel_up_spk = tower_ref_mel_up_spk

		self.all_vars = tf.trainable_variables()

		log('Initialized Tacotron model. Dimensions (? = dynamic shape): ')
		log('  Train mode:               {}'.format(is_training))
		log('  Eval mode:                {}'.format(is_evaluating))
		log('  GTA mode:                 {}'.format(gta))
		log('  Synthesis mode:           {}'.format(not (is_training or is_evaluating)))
		log('  Input:                    {}'.format(inputs.shape))
		for i in range(hp.tacotron_num_gpus):
			log('  device:                   {}'.format(i))
			log('  embedding:                {}'.format(self.tower_embedded_inputs[i].shape))
			log('  enc conv out:             {}'.format(tower_enc_conv_output_shape[i]))
			log('  style_embeddings out:     {}'.format(self.tower_style_embeddings[i].shape))
			log('  encoder out:              {}'.format(self.tower_encoder_outputs[i].shape))
			log('  decoder out:              {}'.format(self.tower_decoder_output[i].shape))
			log('  residual out:             {}'.format(tower_residual[i].shape))
			log('  projected residual out:   {}'.format(tower_projected_residual[i].shape))
			log('  mel out:                  {}'.format(self.tower_mel_outputs[i].shape))
			if post_condition:
				log('  linear out:               {}'.format(self.tower_linear_outputs[i].shape))
			log('  <stop_token> out:         {}'.format(self.tower_stop_token_prediction[i].shape))

		#1_000_000 is causing syntax problems for some people?! Python please :)
		log('  Tacotron Parameters       {:.3f} Million.'.format(np.sum([np.prod(v.get_shape().as_list()) for v in self.all_vars]) / 1000000))


	def add_loss(self):
		'''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
		hp = self._hparams

		self.tower_before_loss = []
		self.tower_after_loss= []
		self.tower_stop_token_loss = []
		self.tower_regularization_loss = []
		self.tower_linear_loss = []
		self.tower_style_emb_loss_emt = []
		self.tower_style_emb_loss_spk = []
		self.tower_style_emb_orthog_loss = []
		self.tower_style_emb_loss_up_emt = []
		self.tower_style_emb_loss_up_spk = []
		self.tower_style_emb_loss_mel_out_up_emt = []
		self.tower_style_emb_loss_mel_out_up_spk = []

		self.tower_loss_no_mo_up = []
		self.tower_loss = []

		if self.args.nat_gan:
			self.tower_d_loss = []
			self.tower_d_loss_targ = []
			self.tower_d_loss_p = []
			self.tower_d_loss_up = []
			self.tower_g_loss_p = []
			self.tower_g_loss_up = []


		total_before_loss = 0
		total_after_loss= 0
		total_stop_token_loss = 0
		total_regularization_loss = 0
		total_linear_loss = 0
		total_style_emb_loss_emt = 0
		total_style_emb_loss_spk = 0
		total_style_emb_orthog_loss = 0
		total_style_emb_loss_up_emt = 0
		total_style_emb_loss_up_spk = 0
		total_style_emb_loss_mel_out_up_emt = 0
		total_style_emb_loss_mel_out_up_spk = 0

		if self.args.nat_gan:
			total_d_loss = 0
			total_d_loss_targ = 0
			total_d_loss_p = 0
			total_d_loss_up = 0
			total_g_loss_p = 0
			total_g_loss_up = 0

		total_loss_no_mo_up = 0
		total_loss = 0

		gpus = ["/gpu:{}".format(i) for i in range(hp.tacotron_num_gpus)]

		for i in range(hp.tacotron_num_gpus):
			with tf.device(tf.train.replica_device_setter(ps_tasks=1, ps_device="/cpu:0", worker_device=gpus[i])):
				with tf.variable_scope('loss') as scope:
					mel_outputs = self.tower_mel_outputs[i] #self.tower_mel_outputs[i][:self.batch_size,:,:] if self.use_unpaired else self.tower_mel_outputs[i]
					decoder_out = self.tower_decoder_output[i] #self.tower_decoder_output[i][:self.batch_size, :, :] if self.use_unpaired else self.tower_decoder_output[i]
					stop_token_prediction= self.tower_stop_token_prediction[i] #self.tower_stop_token_prediction[i][:self.batch_size, :] if self.use_unpaired else self.tower_stop_token_prediction[i]

					if hp.mask_decoder:
						before = MaskedMSE(self.tower_mel_targets[i], decoder_out, self.tower_targets_lengths[i], hparams=self._hparams)
						# Compute loss after postnet
						after = MaskedMSE(self.tower_mel_targets[i], mel_outputs, self.tower_targets_lengths[i], hparams=self._hparams)
						# Compute <stop_token> loss (for learning dynamic generation stop)
						stop_token_loss = MaskedSigmoidCrossEntropy(self.tower_stop_token_targets[i], stop_token_prediction,
																												self.tower_targets_lengths[i], hparams=self._hparams)
						# Compute masked linear loss
						if hp.predict_linear:
							# Compute Linear L1 mask loss (priority to low frequencies)
							linear_loss = MaskedLinearLoss(self.tower_linear_targets[i], self.tower_linear_outputs[i],
																						 self.targets_lengths, hparams=self._hparams)
						else:
							linear_loss=0.
					else:
						# Compute loss of predictions before postnet
						before = tf.losses.mean_squared_error(self.tower_mel_targets[i], decoder_out)
						# Compute loss after postnet
						after = tf.losses.mean_squared_error(self.tower_mel_targets[i], mel_outputs)
						#Compute <stop_token> loss (for learning dynamic generation stop)
						stop_token_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.tower_stop_token_targets[i],
																																										 logits=stop_token_prediction))
						#Linear Loss, not currently used
						if hp.predict_linear:
							#Compute linear loss
							#From https://github.com/keithito/tacotron/blob/tacotron2-work-in-progress/models/tacotron.py
							#Prioritize loss for frequencies under 2000 Hz.
							l1 = tf.abs(self.tower_linear_targets[i] - self.tower_linear_outputs[i])
							n_priority_freq = int(2000 / (hp.sample_rate * 0.5) * hp.num_freq)
							linear_loss = 0.5 * tf.reduce_mean(l1) + 0.5 * tf.reduce_mean(l1[:,:,0:n_priority_freq])
						else:
							linear_loss = 0.

					#Style embedding losses
					style_emb_loss_up_emt=tf.constant(0.)
					style_emb_loss_mel_out_up_emt=tf.constant(0.)
					style_emb_loss_up_spk=tf.constant(0.)
					style_emb_loss_mel_out_up_spk=tf.constant(0.)

					emt_labels_one_hot = tf.one_hot(tf.to_int32(self.tower_emt_labels[i]), self.n_emt)
					style_emb_loss_emt = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.tower_style_emb_logit_emt[i],labels=emt_labels_one_hot))

					spk_labels_one_hot = tf.one_hot(tf.to_int32(self.tower_spk_labels[i]), self.n_spk)
					style_emb_loss_spk = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.tower_style_emb_logit_spk[i],labels=spk_labels_one_hot))

					if self.use_unpaired:
						#Style embedding losses for both unpaired input reference and mel output
						emt_up_labels_one_hot = tf.one_hot(tf.to_int32(self.tower_emt_up_labels[i]), self.n_emt)
						style_emb_loss_up_emt = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.tower_style_emb_logit_up_emt[i], labels=emt_up_labels_one_hot))
						style_emb_loss_mel_out_up_emt = self.args.unpaired_loss_derate * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.tower_style_emb_logit_mel_out_up_emt[i], labels=emt_up_labels_one_hot))

						spk_up_labels_one_hot = tf.one_hot(tf.to_int32(self.tower_spk_up_labels[i]), self.n_spk)
						style_emb_loss_up_spk = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.tower_style_emb_logit_up_spk[i], labels=spk_up_labels_one_hot))
						style_emb_loss_mel_out_up_spk = self.args.unpaired_loss_derate * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.tower_style_emb_logit_mel_out_up_spk[i],labels=spk_up_labels_one_hot))

					# Orthogonal Loss
					if hp.tacotron_use_orthog_loss:
						style_emb_orthog_loss = tf.tensordot(self.tower_refnet_out_emt[i],tf.transpose(self.tower_refnet_out_spk[i]),1)
						style_emb_orthog_loss = .02 * tf.norm(style_emb_orthog_loss) #paper uses squared frobenius, think mistake, just use frobenius
						if self.use_unpaired:
							style_emb_orthog_loss_up = tf.tensordot(self.tower_refnet_out_up_emt[i],tf.transpose(self.tower_refnet_out_up_spk[i]), 1)
							style_emb_orthog_loss += .02 * tf.norm(style_emb_orthog_loss_up)
					else:
						style_emb_orthog_loss = 0.

					# Compute the regularization weight
					if hp.tacotron_scale_regularization:
						reg_weight_scaler = 1. / (2 * hp.max_abs_value) if hp.symmetric_mels else 1. / (hp.max_abs_value)
						reg_weight = hp.tacotron_reg_weight * reg_weight_scaler
					else:
						reg_weight = hp.tacotron_reg_weight

					# Regularize variables
					# Exclude all types of bias, RNN (Bengio et al. On the difficulty of training recurrent neural networks), embeddings and prediction projection layers.
					# Note that we consider attention mechanism v_a weights as a prediction projection layer and we don't regularize it. (This gave better stability)
					regularization = tf.add_n([tf.nn.l2_loss(v) for v in self.all_vars
						if not('bias' in v.name or 'Bias' in v.name or '_projection' in v.name or 'inputs_embedding' in v.name
							or 'RNN' in v.name or 'LSTM' in v.name)]) * reg_weight

					if self.args.nat_gan:
						d_loss_targ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.tower_nat_gan_logits_targets[i], labels=tf.one_hot(tf.constant(0, shape=[self.batch_size_int]),3)))
						d_loss_p = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.tower_nat_gan_logits_mel_p[i], labels=tf.one_hot(tf.constant(1, shape=[self.batch_size_int]),3)))
						d_loss_up = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.tower_nat_gan_logits_mel_up[i], labels=tf.one_hot(tf.constant(2, shape=[self.batch_size_int]),3))) if self.use_unpaired else tf.constant(0.)

						g_loss_p = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.tower_nat_gan_logits_mel_p[i],labels=tf.one_hot(tf.constant(0, shape=[self.batch_size_int]),3)))
						g_loss_up = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.tower_nat_gan_logits_mel_up[i],labels=tf.one_hot(tf.constant(0, shape=[self.batch_size_int]),3))) if self.use_unpaired else tf.constant(0.)

						d_loss = d_loss_targ + d_loss_p + d_loss_up
						g_loss = g_loss_p + g_loss_up
					else:
						g_loss = tf.constant(0.)

					# Compute final loss term
					self.tower_before_loss.append(before)
					self.tower_after_loss.append(after)
					self.tower_stop_token_loss.append(stop_token_loss)
					self.tower_regularization_loss.append(regularization)
					self.tower_linear_loss.append(linear_loss)
					self.tower_style_emb_orthog_loss.append(style_emb_orthog_loss)

					self.tower_style_emb_loss_emt.append(style_emb_loss_emt)
					self.tower_style_emb_loss_emt.append(style_emb_loss_spk)
					self.tower_style_emb_loss_up_emt.append(style_emb_loss_up_emt)
					self.tower_style_emb_loss_up_spk.append(style_emb_loss_up_spk)
					self.tower_style_emb_loss_mel_out_up_emt.append(style_emb_loss_mel_out_up_emt)
					self.tower_style_emb_loss_mel_out_up_spk.append(style_emb_loss_mel_out_up_spk)

					if self.args.nat_gan:
						self.tower_d_loss.append(d_loss)
						self.tower_d_loss_targ.append(d_loss_targ)
						self.tower_d_loss_p.append(d_loss_p)
						self.tower_d_loss_up.append(d_loss_up)
						self.tower_g_loss_p.append(g_loss_p)
						self.tower_g_loss_up.append(g_loss_up)

					# Tower loss without the mel_output reference embedder
					tower_loss_no_mo_up = before + after + stop_token_loss + regularization + linear_loss +\
											 style_emb_loss_emt + style_emb_loss_spk + style_emb_orthog_loss + style_emb_loss_up_emt +\
											 style_emb_loss_up_spk + g_loss

					# All losses including mel_output reference embedder
					tower_loss = tower_loss_no_mo_up + style_emb_loss_mel_out_up_emt + style_emb_loss_mel_out_up_spk

					self.tower_loss_no_mo_up.append(tower_loss_no_mo_up)
					self.tower_loss.append(tower_loss)

			total_before_loss += before
			total_after_loss += after
			total_stop_token_loss += stop_token_loss
			total_regularization_loss += regularization
			total_linear_loss += linear_loss
			total_style_emb_loss_emt += style_emb_loss_emt
			total_style_emb_loss_spk += style_emb_loss_spk
			total_style_emb_orthog_loss += style_emb_orthog_loss
			total_style_emb_loss_up_emt += style_emb_loss_up_emt
			total_style_emb_loss_up_spk += style_emb_loss_up_spk
			total_style_emb_loss_mel_out_up_emt += style_emb_loss_mel_out_up_emt
			total_style_emb_loss_mel_out_up_spk += style_emb_loss_mel_out_up_spk

			if self.args.nat_gan:
				total_d_loss+=d_loss
				total_d_loss_targ+=d_loss_targ
				total_d_loss_p+=d_loss_p
				total_d_loss_up+=d_loss_up
				total_g_loss_p+=g_loss_p
				total_g_loss_up+=g_loss_up

			total_loss_no_mo_up += tower_loss_no_mo_up

			total_loss += tower_loss

		self.before_loss = total_before_loss / hp.tacotron_num_gpus
		self.after_loss = total_after_loss / hp.tacotron_num_gpus
		self.stop_token_loss = total_stop_token_loss / hp.tacotron_num_gpus
		self.regularization_loss = total_regularization_loss / hp.tacotron_num_gpus
		self.linear_loss = total_linear_loss / hp.tacotron_num_gpus
		self.style_emb_loss_emt = total_style_emb_loss_emt / hp.tacotron_num_gpus
		self.style_emb_loss_spk = total_style_emb_loss_spk / hp.tacotron_num_gpus
		self.style_emb_orthog_loss = total_style_emb_orthog_loss / hp.tacotron_num_gpus
		self.style_emb_loss_up_emt = total_style_emb_loss_up_emt / hp.tacotron_num_gpus
		self.style_emb_loss_up_spk = total_style_emb_loss_up_spk / hp.tacotron_num_gpus
		self.style_emb_loss_mel_out_up_emt = total_style_emb_loss_mel_out_up_emt / hp.tacotron_num_gpus
		self.style_emb_loss_mel_out_up_spk = total_style_emb_loss_mel_out_up_spk / hp.tacotron_num_gpus
		self.loss_no_mo_up = total_loss_no_mo_up / hp.tacotron_num_gpus

		self.loss = total_loss / hp.tacotron_num_gpus

		if self.args.nat_gan:
			self.d_loss = total_d_loss / hp.tacotron_num_gpus
			self.d_loss_targ = total_d_loss_targ / hp.tacotron_num_gpus
			self.d_loss_p = total_d_loss_p / hp.tacotron_num_gpus
			self.g_loss_p = total_g_loss_p / hp.tacotron_num_gpus
			self.d_loss_up = total_d_loss_up / hp.tacotron_num_gpus
			self.g_loss_up = total_g_loss_up / hp.tacotron_num_gpus

	def add_optimizer(self, global_step):
		'''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.
		Args:
			global_step: int32 scalar Tensor representing current global step in training
		'''
		hp = self._hparams
		tower_gradients = []

		#used to optimize reference encoders without using mel output style embedding loss
		tower_gradients_r = []

		# 1. Declare GPU Devices
		gpus = ["/gpu:{}".format(i) for i in range(hp.tacotron_num_gpus)]

		grad_device = '/cpu:0' if hp.tacotron_num_gpus > 1 else gpus[0]

		with tf.device(grad_device):
			if hp.tacotron_decay_learning_rate:
				self.decay_steps = hp.tacotron_decay_steps
				self.decay_rate = hp.tacotron_decay_rate
				self.learning_rate = self._learning_rate_decay(hp.tacotron_initial_learning_rate, global_step)
			else:
				self.learning_rate = tf.convert_to_tensor(hp.tacotron_initial_learning_rate)

			with tf.variable_scope('optimizer') as scope:
				optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.tacotron_adam_beta1, hp.tacotron_adam_beta2,
																					 hp.tacotron_adam_epsilon)

			#not using right now
			if False:#self.args.unpaired:
				with tf.variable_scope('optimizer_r') as scope:
					optimizer_r = tf.train.AdamOptimizer(self.learning_rate, hp.tacotron_adam_beta1, hp.tacotron_adam_beta2,
																								 hp.tacotron_adam_epsilon)

		# 2. Compute Gradient
		for i in range(hp.tacotron_num_gpus):
			#  Device placement
			with tf.device(tf.train.replica_device_setter(ps_tasks=1, ps_device="/cpu:0", worker_device=gpus[i])):
				with tf.variable_scope('optimizer') as scope:
					# update_vars = [v for v in self.all_vars if not ('refnet' in v.name or 'style_disc' in v.name)] if self.args.unpaired else self.all_vars
					update_vars = self.all_vars
					update_vars = [v for v in update_vars if not ('inputs_embedding' in v.name or 'encoder_' in v.name)] if hp.tacotron_fine_tuning else update_vars

					gradients = optimizer.compute_gradients(self.tower_loss[i], var_list=update_vars) #self.tower_loss_no_mo_up[i], var_list=update_vars)
					tower_gradients.append(gradients)

			# not using right now
			if False:#self.args.unpaired:
				with tf.variable_scope('optimizer_r') as scope:
					update_vars_r = [v for v in self.all_vars if ('refnet' in v.name or 'style_disc' in v.name)]
					gradients_r = optimizer_r.compute_gradients(self.tower_loss_no_mo_up[i], var_list=update_vars_r)
					tower_gradients_r.append(gradients_r)

					# print("\nREFNET VARS")
					# for v in update_vars_r:
					# 	print("refnet", v)

		# 3. Average Gradient
		with tf.device(grad_device):
			with tf.variable_scope('optimizer') as scope:
				clipped_gradients, variables = self.get_clipped_grads(tower_gradients, hp)

				# Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
				# https://github.com/tensorflow/tensorflow/issues/1122
				with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
					self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables), global_step=global_step)

			# not using right now
			if False:#self.args.unpaired:
				with tf.variable_scope('optimizer_r') as scope:
					clipped_gradients_r, variables_r = self.get_clipped_grads(tower_gradients_r, hp, GEN=False)

					# Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
					# https://github.com/tensorflow/tensorflow/issues/1122
					with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
						self.optimize_r = optimizer_r.apply_gradients(zip(clipped_gradients_r, variables_r), global_step=global_step)


	# def add_optimizer(self, global_step):
	# 	'''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.
	# 	Args:
	# 		global_step: int32 scalar Tensor representing current global step in training
	# 	'''
	# 	hp = self._hparams
	# 	tower_gradients = []
	# 	tower_gradients_d = []
	#
	# 	# 1. Declare GPU Devices
	# 	gpus = ["/gpu:{}".format(i) for i in range(hp.tacotron_num_gpus)]
	#
	# 	grad_device = '/cpu:0' if hp.tacotron_num_gpus > 1 else gpus[0]
	#
	# 	with tf.device(grad_device):
	# 		if hp.tacotron_decay_learning_rate:
	# 			self.decay_steps = hp.tacotron_decay_steps
	# 			self.decay_rate = hp.tacotron_decay_rate
	# 			self.learning_rate = self._learning_rate_decay(hp.tacotron_initial_learning_rate, global_step)
	# 		else:
	# 			self.learning_rate = tf.convert_to_tensor(hp.tacotron_initial_learning_rate)
	#
	# 		with tf.variable_scope('optimizer') as scope:
	# 			optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.tacotron_adam_beta1, hp.tacotron_adam_beta2, hp.tacotron_adam_epsilon)
	#
	# 		if self.args.nat_gan:
	# 			##NATURALNESS_GAN DISCRIMINATOR SCOPE
	# 			with tf.variable_scope('optimizer_d') as scope:
	# 				optimizer_d = tf.train.AdamOptimizer(self.learning_rate, hp.tacotron_adam_beta1, hp.tacotron_adam_beta2, hp.tacotron_adam_epsilon)
	#
	# 	# 2. Compute Gradient
	# 	for i in range(hp.tacotron_num_gpus):
	# 		#  Device placement
	# 		with tf.device(tf.train.replica_device_setter(ps_tasks=1, ps_device="/cpu:0", worker_device=gpus[i])):
	# 			with tf.variable_scope('optimizer') as scope:
	# 				# update_vars = [v for v in self.all_vars if not ('inputs_embedding' in v.name or 'encoder_' in v.name)] if hp.tacotron_fine_tuning else None
	#
	# 				#remove naturalness gan discrimnator from generator optimizer
	# 				update_vars = [v for v in self.all_vars if not ('nat_gan' in v.name)]
	#
	# 				#allow for fine-tuning (don't update embedding encoder)
	# 				update_vars = [v for v in update_vars if not ('inputs_embedding' in v.name or 'encoder_' in v.name)] if hp.tacotron_fine_tuning else update_vars
	#
	# 				if self.args.lock_ref_enc:
	# 					var_list = update_vars if update_vars != None else self.all_vars
	# 					update_vars  = [v for v in var_list if not ('refnet' in v.name)]
	# 					if self._hparams.tacotron_style_emb_disc_refnet:
	# 						update_vars = [v for v in update_vars if not ('style_disc' in v.name)]
	# 				if self.args.lock_gst:
	# 					var_list = update_vars if update_vars != None else self.all_vars
	# 					update_vars  = [v for v in var_list if not ('Multihead-attention' in v.name or 'style_tokens' in v.name)]
	# 					if not(self._hparams.tacotron_style_emb_disc_refnet):
	# 						update_vars = [v for v in update_vars if not ('style_disc' in v.name)]
	#
	# 				gradients = optimizer.compute_gradients(self.tower_loss[i], var_list=update_vars)
	# 				tower_gradients.append(gradients)
	#
	# 		if self.args.nat_gan:
	# 			with tf.variable_scope('optimizer_d') as scope:
	# 				# only naturalness gan discrimnator from generator optimizer
	# 				update_vars_d = [v for v in self.all_vars if ('nat_gan' in v.name)]
	# 				gradients_d = optimizer_d.compute_gradients(self.tower_d_loss[i], var_list=update_vars_d)
	# 				tower_gradients_d.append(gradients_d)
	#
	# 	# 3. Average Gradient
	# 	with tf.device(grad_device):
	# 		with tf.variable_scope('optimizer') as scope:
	# 			clipped_gradients, variables = self.get_clipped_grads(tower_gradients, hp)
	#
	# 			# Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
	# 			# https://github.com/tensorflow/tensorflow/issues/1122
	# 			with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
	# 				self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables), global_step=global_step)
	#
	# 		if self.args.nat_gan:
	# 			with tf.variable_scope('optimizer_d') as scope:
	# 				clipped_gradients_d, variables_d = self.get_clipped_grads(tower_gradients_d, hp, GEN=False)
	#
	# 				# Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
	# 				# https://github.com/tensorflow/tensorflow/issues/1122
	# 				with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
	# 					self.optimize_d = optimizer_d.apply_gradients(zip(clipped_gradients_d, variables_d), global_step=global_step)

	def get_clipped_grads(self, tower_gradients, hp, GEN=True):

		avg_grads = []
		variables = []
		for grad_and_vars in zip(*tower_gradients):
			# each_grads_vars = ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
			grads = []
			for g, v in grad_and_vars:
				expanded_g = tf.expand_dims(g, 0)
				# Append on a 'tower' dimension which we will average over below.
				grads.append(expanded_g)
			# Average over the 'tower' dimension.

			grad = tf.concat(axis=0, values=grads)
			grad = tf.reduce_mean(grad, 0)

			v = grad_and_vars[0][1]
			avg_grads.append(grad)
			variables.append(v)

		if GEN:
			self.gradients = avg_grads
		else:
			self.gradients_d = avg_grads
		# Just for caution
		# https://github.com/Rayhane-mamah/Tacotron-2/issues/11
		if hp.tacotron_clip_gradients:
			clipped_gradients, _ = tf.clip_by_global_norm(avg_grads, 1.)  # __mark 0.5 refer
		else:
			clipped_gradients = avg_grads

		return(clipped_gradients, variables)

	def _learning_rate_decay(self, init_lr, global_step):
		#################################################################
		# Narrow Exponential Decay:

		# Phase 1: lr = 1e-3
		# We only start learning rate decay after 50k steps

		# Phase 2: lr in ]1e-5, 1e-3[
		# decay reach minimal value at step 310k

		# Phase 3: lr = 1e-5
		# clip by minimal learning rate value (step > 310k)
		#################################################################
		hp = self._hparams

		#Compute natural exponential decay
		lr = tf.train.exponential_decay(init_lr, 
			global_step - hp.tacotron_start_decay, #lr = 1e-3 at step 50k
			self.decay_steps, 
			self.decay_rate, #lr = 1e-5 around step 310k
			name='lr_exponential_decay')


		#clip learning rate by max and min values (initial and final values)
		return tf.minimum(tf.maximum(lr, hp.tacotron_final_learning_rate), init_lr)