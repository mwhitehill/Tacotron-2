import os
import wave
from datetime import datetime
import platform
import time

import numpy as np
import pyaudio
import sounddevice as sd
import tensorflow as tf
from datasets import audio
from infolog import log
from librosa import effects
from tacotron.models import create_model
from tacotron.utils import plot
from tacotron.utils.text import text_to_sequence


class Synthesizer:
	def load(self, args,checkpoint_path, hparams, gta=False, use_intercross=False, n_emt=4, n_spk=2):

		self.args = args

		model_name = 'Tacotron_emt_attn' if args.emt_attn else 'Tacotron'
		log('Constructing model: %s' % model_name)
		#Force the batch size to be known in order to use attention masking in batch synthesis
		inputs = tf.placeholder(tf.int32, (None, None), name='inputs')
		input_lengths = tf.placeholder(tf.int32, (None), name='input_lengths')
		targets = tf.placeholder(tf.float32, (None, None, hparams.num_mels), name='mel_targets')
		split_infos = tf.placeholder(tf.int32, shape=(hparams.tacotron_num_gpus, None), name='split_infos')

		mel_refs_emt = tf.placeholder(tf.float32, (None, None, hparams.num_mels), name='mel_refs_emt')
		mel_refs_spk = tf.placeholder(tf.float32, (None, None, hparams.num_mels), name='mel_refs_spk')

		emt_labels_synth = tf.placeholder(tf.int32, (None,), name='emt_labels_synth')
		spk_labels_synth = tf.placeholder(tf.int32, (None,), name='spk_labels_synth')

		with tf.variable_scope('Tacotron_model', reuse=tf.AUTO_REUSE) as scope:
			self.model = create_model(model_name, hparams)
			if gta:
				self.model.initialize(args, inputs, input_lengths, targets, gta=gta, split_infos=split_infos, use_intercross=use_intercross,
															ref_mel_emt=mel_refs_emt, ref_mel_spk=mel_refs_spk, synth=True,
															emt_labels=emt_labels_synth, spk_labels=spk_labels_synth)
			else:
				self.model.initialize(args, inputs, input_lengths, split_infos=split_infos, use_intercross=use_intercross,
															ref_mel_emt=mel_refs_emt, ref_mel_spk=mel_refs_spk, n_emt=4, n_spk=2, synth=True,
															emt_labels=emt_labels_synth, spk_labels=spk_labels_synth)

			self.mel_outputs = self.model.tower_mel_outputs
			self.linear_outputs = self.model.tower_linear_outputs if (hparams.predict_linear and not gta) else None
			self.alignments = self.model.tower_alignments
			self.stop_token_prediction = self.model.tower_stop_token_prediction
			self.targets = targets

		if hparams.GL_on_GPU:
			self.GLGPU_mel_inputs = tf.placeholder(tf.float32, (None, hparams.num_mels), name='GLGPU_mel_inputs')
			self.GLGPU_lin_inputs = tf.placeholder(tf.float32, (None, hparams.num_freq), name='GLGPU_lin_inputs')

			self.GLGPU_mel_outputs = audio.inv_mel_spectrogram_tensorflow(self.GLGPU_mel_inputs, hparams)
			self.GLGPU_lin_outputs = audio.inv_linear_spectrogram_tensorflow(self.GLGPU_lin_inputs, hparams)

		self.gta = gta
		self._hparams = hparams
		#pad input sequences with the <pad_token> 0 ( _ )
		self._pad = 0
		#explicitely setting the padding to a value that doesn't originally exist in the spectogram
		#to avoid any possible conflicts, without affecting the output range of the model too much
		if hparams.symmetric_mels:
			self._target_pad = -hparams.max_abs_value
		else:
			self._target_pad = 0.

		self.inputs = inputs
		self.input_lengths = input_lengths
		self.targets = targets
		self.split_infos = split_infos

		self.mel_refs_emt = mel_refs_emt
		self.mel_refs_spk = mel_refs_spk

		self.emt_labels = emt_labels_synth
		self.spk_labels = spk_labels_synth

		log('Loading checkpoint: %s' % checkpoint_path)
		#Memory allocation on the GPUs as needed
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.allow_soft_placement = True

		self.session = tf.Session(config=config)
		self.session.run(tf.global_variables_initializer())

		saver = tf.train.Saver()
		saver.restore(self.session, checkpoint_path)


	def synthesize(self, texts, basenames, out_dir, log_dir, mel_filenames, basenames_refs=None,
								 mel_ref_filenames_emt=None, mel_ref_filenames_spk=None, emb_only=False,
								 emt_labels_synth=None, spk_labels_synth=None):

		hparams = self._hparams
		cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
		#[-max, max] or [0,max]
		T2_output_range = (-hparams.max_abs_value, hparams.max_abs_value) if hparams.symmetric_mels else (0, hparams.max_abs_value)

		#Repeat last sample until number of samples is dividable by the number of GPUs (last run scenario)
		while len(texts) % hparams.tacotron_synthesis_batch_size != 0:
			texts.append(texts[-1])
			basenames.append(basenames[-1])
			basenames_refs.append(basenames_refs[-1])
			if mel_filenames is not None:
				mel_filenames.append(mel_filenames[-1])
			if mel_ref_filenames_emt is not None:
				mel_ref_filenames_emt.append(mel_ref_filenames_emt[-1])
			if mel_ref_filenames_spk is not None:
				mel_ref_filenames_spk.append(mel_ref_filenames_spk[-1])
			if emt_labels_synth is not None:
				emt_labels_synth.append(emt_labels_synth[-1])
			if spk_labels_synth is not None:
				spk_labels_synth.append(spk_labels_synth[-1])

		assert 0 == len(texts) % self._hparams.tacotron_num_gpus
		seqs = [np.asarray(text_to_sequence(text, cleaner_names)) for text in texts]
		input_lengths = [len(seq) for seq in seqs]
		size_per_device = len(seqs) // self._hparams.tacotron_num_gpus

		#Pad inputs according to each GPU max length
		input_seqs = None
		split_infos = []

		np_mel_refs_emt = [np.load(f) for f in mel_ref_filenames_emt]
		np_mel_refs_spk = [np.load(f) for f in mel_ref_filenames_spk]

		mel_ref_seqs_emt = None
		mel_ref_seqs_spk = None

		for i in range(self._hparams.tacotron_num_gpus):
			device_input = seqs[size_per_device*i: size_per_device*(i+1)]
			device_input, max_seq_len = self._prepare_inputs(device_input)

			input_seqs = np.concatenate((input_seqs, device_input), axis=1) if input_seqs is not None else device_input

			device_mel_ref_emt = np_mel_refs_emt[size_per_device * i: size_per_device * (i + 1)]
			device_mel_ref_emt, max_mel_ref_len_emt = self._prepare_targets(device_mel_ref_emt, self._hparams.outputs_per_step)
			mel_ref_seqs_emt = np.concatenate((mel_ref_seqs_emt, device_mel_ref_emt), axis=1) if mel_ref_seqs_emt is not None else device_mel_ref_emt

			device_mel_ref_spk = np_mel_refs_spk[size_per_device * i: size_per_device * (i + 1)]
			device_mel_ref_spk, max_mel_ref_len_spk = self._prepare_targets(device_mel_ref_spk, self._hparams.outputs_per_step)
			mel_ref_seqs_spk = np.concatenate((mel_ref_seqs_spk, device_mel_ref_spk), axis=1) if mel_ref_seqs_spk is not None else device_mel_ref_spk

			split_infos.append([max_seq_len, 0, 0, 0, 0, max_mel_ref_len_emt, max_mel_ref_len_spk])

		feed_dict = {
			self.inputs: input_seqs,
			self.input_lengths: np.asarray(input_lengths, dtype=np.int32),
			self.mel_refs_emt: mel_ref_seqs_emt,
			self.mel_refs_spk: mel_ref_seqs_spk,
			self.spk_labels: np.asarray(spk_labels_synth, dtype=np.int32),
			self.emt_labels: np.asarray(emt_labels_synth, dtype=np.int32)
		}

		if self.gta:
			np_targets = [np.load(mel_filename) for mel_filename in mel_filenames]
			target_lengths = [len(np_target) for np_target in np_targets]

			#pad targets according to each GPU max length
			target_seqs = None
			for i in range(self._hparams.tacotron_num_gpus):
				device_target = np_targets[size_per_device*i: size_per_device*(i+1)]
				device_target, max_target_len = self._prepare_targets(device_target, self._hparams.outputs_per_step)
				target_seqs = np.concatenate((target_seqs, device_target), axis=1) if target_seqs is not None else device_target
				split_infos[i][1] = max_target_len #Not really used but setting it in case for future development maybe?

			feed_dict[self.targets] = target_seqs
			assert len(np_targets) == len(texts)

		feed_dict[self.split_infos] = np.asarray(split_infos, dtype=np.int32)

		if emb_only:
			if self.args.emt_attn:
				return(self.session.run([self.model.tower_refnet_out_emt[0],
															 self.model.tower_refnet_out_spk[0],
															 self.model.tower_refnet_outputs_mel_out_emt[0],
															 self.model.tower_refnet_outputs_mel_out_spk[0],
															 self.model.tower_context_emt[0]],feed_dict=feed_dict))
			else:
				return (self.session.run([self.model.tower_refnet_out_emt[0],
																	self.model.tower_refnet_out_spk[0],
																	self.model.tower_refnet_outputs_mel_out_emt[0],
																	self.model.tower_refnet_outputs_mel_out_spk[0],
																	tf.constant(1.)], feed_dict=feed_dict))

		if self.gta or not hparams.predict_linear:
			if self.args.attn == 'style_tokens':
				mels, alignments, stop_tokens = self.session.run([self.mel_outputs,
																										self.alignments,
																										self.stop_token_prediction],
																									 feed_dict=feed_dict)
			else:
				mels, alignments, stop_tokens, refnet_emt,\
				ref_emt, alignments_emt = self.session.run([self.mel_outputs,
																													self.alignments,
																													self.stop_token_prediction,
																													self.model.tower_refnet_out_emt[0],
																													self.model.tower_ref_mel_emt[0],
																													self.model.tower_alignments_emt],
																													#self.model.tower_context_emt[0],
																													#self.model.tower_refnet_out_spk[0]],
																										feed_dict=feed_dict)

			# import pandas as pd
			# df_cont = pd.DataFrame(cont[0])
			# df_cont.to_csv(r'C:\Users\t-mawhit\Documents\code\Tacotron-2\eval\test\cont.csv')
			# pd.DataFrame(refnet_spk).to_csv(r'C:\Users\t-mawhit\Documents\code\Tacotron-2\eval\test\r_spk.csv')
			# raise

			# print(refnet_emt[:,0:5])
			# print(refnet_spk[:,0:5])
			# for i,(m1,m2,m3) in enumerate(zip(mels[0],ref_emt,ref_spk)):
			# 	np.save('../eval/mels_save/{}_mel.npy'.format(i),m1)
			# 	np.save('../eval/mels_save/{}_ref_emt.npy'.format(i), m2)
			# 	np.save('../eval/mels_save/{}_ref_spk.npy'.format(i), m3)
			# time.sleep(.5)
			# raise

			#Linearize outputs (n_gpus -> 1D)
			mels = [mel for gpu_mels in mels for mel in gpu_mels]
			alignments = [align for gpu_aligns in alignments for align in gpu_aligns]
			stop_tokens = [token for gpu_token in stop_tokens for token in gpu_token]
			if self.args.emt_attn and not(self.args.attn == 'style_tokens'):
				alignments_emt = [align_emt for gpu_aligns_emt in alignments_emt for align_emt in gpu_aligns_emt]

			if not self.gta:
				#Natural batch synthesis
				#Get Mel lengths for the entire batch from stop_tokens predictions
				target_lengths = self._get_output_lengths(stop_tokens)

			#Take off the batch wise padding
			mels = [mel[:target_length, :] for mel, target_length in zip(mels, target_lengths)]
			assert len(mels) == len(texts)

		else:
			linears, mels, alignments, stop_tokens = self.session.run([self.linear_outputs, self.mel_outputs,
																																 self.alignments, self.stop_token_prediction], feed_dict=feed_dict)

			#Linearize outputs (1D arrays)
			linears = [linear for gpu_linear in linears for linear in gpu_linear]
			mels = [mel for gpu_mels in mels for mel in gpu_mels]
			alignments = [align for gpu_aligns in alignments for align in gpu_aligns]
			stop_tokens = [token for gpu_token in stop_tokens for token in gpu_token]

			#Natural batch synthesis
			#Get Mel/Linear lengths for the entire batch from stop_tokens predictions
			target_lengths = self._get_output_lengths(stop_tokens)

			#Take off the batch wise padding
			mels = [mel[:target_length, :] for mel, target_length in zip(mels, target_lengths)]
			linears = [linear[:target_length, :] for linear, target_length in zip(linears, target_lengths)]
			linears = np.clip(linears, T2_output_range[0], T2_output_range[1])
			assert len(mels) == len(linears) == len(texts)

		mels = [np.clip(m, T2_output_range[0], T2_output_range[1]) for m in mels]

		if basenames is None:
			#Generate wav and read it
			if hparams.GL_on_GPU:
				wav = self.session.run(self.GLGPU_mel_outputs, feed_dict={self.GLGPU_mel_inputs: mels[0]})
				wav = audio.inv_preemphasis(wav, hparams.preemphasis, hparams.preemphasize)
			else:
				wav = audio.inv_mel_spectrogram(mels[0].T, hparams)
			audio.save_wav(wav, 'temp.wav', sr=hparams.sample_rate) #Find a better way

			if platform.system() == 'Linux':
				#Linux wav reader
				os.system('aplay temp.wav')

			elif platform.system() == 'Windows':
				#windows wav reader
				os.system('start /min mplay32 /play /close temp.wav')

			else:
				raise RuntimeError('Your OS type is not supported yet, please add it to "tacotron/synthesizer.py, line-165" and feel free to make a Pull Request ;) Thanks!')

			return


		saved_mels_paths = []
		speaker_ids = []
		for i, mel in enumerate(mels):
			#Get speaker id for global conditioning (only used with GTA generally)
			if hparams.gin_channels > 0:
				raise RuntimeError('Please set the speaker_id rule in line 99 of tacotron/synthesizer.py to allow for global condition usage later.')
				speaker_id = '<no_g>' #set the rule to determine speaker id. By using the file basename maybe? (basenames are inside "basenames" variable)
				speaker_ids.append(speaker_id) #finish by appending the speaker id. (allows for different speakers per batch if your model is multispeaker)
			else:
				speaker_id = '<no_g>'
				speaker_ids.append(speaker_id)

			# Write the spectrogram to disk
			# Note: outputs mel-spectrogram files and target ones have same names, just different folders
			mel_filename = os.path.join(out_dir, 'mel-{}_{}.npy'.format(basenames[i],basenames_refs[i]))
			# np.save(mel_filename, mel, allow_pickle=False)
			saved_mels_paths.append(mel_filename)
			if log_dir is not None:
				os.makedirs(os.path.join(log_dir,'wavs'),exist_ok=True)
				os.makedirs(os.path.join(log_dir, 'plots'),exist_ok=True)
				#save wav (mel -> wav)
				if hparams.GL_on_GPU:
					wav = self.session.run(self.GLGPU_mel_outputs, feed_dict={self.GLGPU_mel_inputs: mel})
					wav = audio.inv_preemphasis(wav, hparams.preemphasis, hparams.preemphasize)
				else:
					wav = audio.inv_mel_spectrogram(mel.T, hparams)

				#add silence to make ending of file more noticeable
				wav = np.append(np.append(np.zeros(int(.5*hparams.sample_rate)), wav),np.zeros(int(.5*hparams.sample_rate)))
				audio.save_wav(wav, os.path.join(log_dir, 'wavs/wav-{}_{}.wav'.format(basenames[i],basenames_refs[i])), sr=hparams.sample_rate)

				#save alignments
				plot.plot_alignment(alignments[i], os.path.join(log_dir, 'plots/alignment-{}_{}.png'.format(basenames[i],basenames_refs[i])),
					title='{}'.format(texts[i]), split_title=True, max_len=target_lengths[i])

				if self.args.emt_attn and self.args.attn == 'simple':
					plot.plot_alignment(alignments_emt[i], os.path.join(log_dir, 'plots/alignment_emt-{}_{}.png'.format(basenames[i],basenames_refs[i])),
						title='{}'.format(texts[i]), split_title=True, max_len=target_lengths[i])

				#save mel spectrogram plot
				plot.plot_spectrogram(mel, os.path.join(log_dir, 'plots/mel-{}_{}.png'.format(basenames[i],basenames_refs[i])),
					title='{}'.format(texts[i]), split_title=True)
				print("Finished saving {}_{}".format(basenames[i],basenames_refs[i]))

				if hparams.predict_linear:
					#save wav (linear -> wav)
					if hparams.GL_on_GPU:
						wav = self.session.run(self.GLGPU_lin_outputs, feed_dict={self.GLGPU_lin_inputs: linears[i]})
						wav = audio.inv_preemphasis(wav, hparams.preemphasis, hparams.preemphasize)
					else:
						wav = audio.inv_linear_spectrogram(linears[i].T, hparams)
					audio.save_wav(wav, os.path.join(log_dir, 'wavs/wav-{}-linear_{}.wav'.format(basenames[i],basenames_refs[i])), sr=hparams.sample_rate)

					#save linear spectrogram plot
					plot.plot_spectrogram(linears[i], os.path.join(log_dir, 'plots/linear-{}_{}.png'.format(basenames[i],basenames_refs[i])),
						title='{}'.format(texts[i]), split_title=True, auto_aspect=True)

		return saved_mels_paths, speaker_ids

	def _round_up(self, x, multiple):
		remainder = x % multiple
		return x if remainder == 0 else x + multiple - remainder

	def _prepare_inputs(self, inputs):
		max_len = max([len(x) for x in inputs])
		return np.stack([self._pad_input(x, max_len) for x in inputs]), max_len

	def _pad_input(self, x, length):
		return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=self._pad)

	def _prepare_targets(self, targets, alignment):
		max_len = max([len(t) for t in targets])
		data_len = self._round_up(max_len, alignment)
		return np.stack([self._pad_target(t, data_len) for t in targets]), data_len

	def _pad_target(self, t, length):
		return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=self._target_pad)

	def _get_output_lengths(self, stop_tokens):
		#Determine each mel length by the stop token predictions. (len = first occurence of 1 in stop_tokens row wise)
		output_lengths = [row.index(1) if 1 in row else len(row) for row in np.round(stop_tokens).tolist()]
		return output_lengths
