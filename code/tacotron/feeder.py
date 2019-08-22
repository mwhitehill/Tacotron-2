import os
import threading
import time
import traceback

import numpy as np
import tensorflow as tf
import pandas as pd

if __name__ == '__main__':
	import sys
	sys.path.append(os.getcwd())
	import argparse

from infolog import log
from sklearn.model_selection import train_test_split
from tacotron.utils.text import text_to_sequence
from hparams import hparams

TEST=False
test_size = (hparams.tacotron_test_size if hparams.tacotron_test_size is not None
			else hparams.tacotron_test_batches * hparams.tacotron_batch_size)

#Not used anymore - generate dataframe from metadata list - allows clipping out short utterances
def get_metadata_df(path, args):

	# load metadata into a dataframe
	columns = ['dataset','audio_filename', 'mel_filename', 'linear_filename', 'spk_emb_filename', 'time_steps', 'mel_frames', 'text',
						 'emt_label', 'spk_label', 'basename', 'sex']
	meta_df = pd.read_csv(path, sep='|')
	meta_df.columns = columns

	if args.remove_long_samps:
		raise ValueError('Should not be using remove long samples with getting metatdata dataframe from text file')
		# idxs = meta_df[meta_df['basename'].map(lambda x: str(x).endswith('_023.wav'))].index.values
		# idxs = np.append(idxs,meta_df[meta_df['basename'].map(lambda x: str(x).endswith('_021.wav'))].index.values)
		# idxs = np.append(idxs, meta_df[meta_df['mel_frames'].map(lambda x: x>=900)].index.values)
		# meta_df = meta_df.drop(idxs)

	return(meta_df)

class Feeder:
	"""
		Feeds batches of data into queue on a background thread.
	"""

	def __init__(self, coordinator, metadata_filename, hparams, args):
		super(Feeder, self).__init__()
		self._coord = coordinator
		self._hparams = hparams
		self._cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
		self._train_offset = 0
		self._test_offset = 0
		self._args = args

		# Load metadata
		self.data_folder = os.path.dirname(metadata_filename)
		if args.test_max_len or args.TEST or TEST:
			self._batches_per_group = 2
		else:
			self._batches_per_group = 64

		with open(metadata_filename, encoding='utf-8') as f:
			self._metadata = [line.strip().split('|') for line in f]
			if args.remove_long_samps:
				len_before = len(self._metadata)
				#remove the 2 longest utterances + any over 900 frames long
				self._metadata = [f for f in self._metadata if not(f[10].endswith('_023.wav'))]
				self._metadata = [f for f in self._metadata if not(f[10].endswith('_021.wav'))]
				self._metadata = [f for f in self._metadata if int(f[6]) < 500]
				print("Removed Long Samples")
				print("# samps before:", len_before)
				print("# samps after:", len(self._metadata))
			frame_shift_ms = hparams.hop_size / hparams.sample_rate
			hours = sum([int(x[6]) for x in self._metadata]) * frame_shift_ms / (3600)
			log('Loaded metadata for {} examples ({:.2f} hours)'.format(len(self._metadata), hours))

		# self._metadata_df = get_metadata_df(metadata_filename, args)
		self._metadata_df = pd.DataFrame(self._metadata)
		columns = ['dataset', 'audio_filename', 'mel_filename', 'linear_filename', 'spk_emb_filename', 'time_steps',
							 'mel_frames', 'text','emt_label', 'spk_label', 'basename', 'sex']
		self._metadata_df.columns = columns

		#Train test split
		if hparams.tacotron_test_size is None:
			assert hparams.tacotron_test_batches is not None

		indices = np.arange(len(self._metadata))
		train_indices, test_indices = train_test_split(indices,
			test_size=test_size, random_state=hparams.tacotron_data_random_state)

		#Make sure test_indices is a multiple of batch_size else round down
		len_test_indices = self._round_down(len(test_indices), hparams.tacotron_batch_size)
		extra_test = test_indices[len_test_indices:]
		test_indices = test_indices[:len_test_indices]
		train_indices = np.concatenate([train_indices, extra_test])

		self._train_meta = list(np.array(self._metadata)[train_indices])
		self._test_meta = list(np.array(self._metadata)[test_indices])

		if args.test_max_len:
			self._train_meta.sort(key=lambda x: int(x[6]),reverse=True)
			self._test_meta.sort(key=lambda x: int(x[6]), reverse=True)
			print("TESTING MAX LENGTH FOR SAMPLES TO FIND MAX BATCH SIZE")

		self._metadata_df['train_test'] = 'train'
		self._metadata_df.iloc[np.array(sorted(test_indices))-1,-1] = 'test'
		self.df_meta_train = self._metadata_df[self._metadata_df.loc[:,'train_test'] == 'train']

		self.total_emt = self._metadata_df.emt_label.unique()
		self.total_spk = self._metadata_df.spk_label.unique()

		self.test_steps = args.tacotron_test_steps #len(self._test_meta) // hparams.tacotron_batch_size

		if hparams.tacotron_test_size is None:
			assert hparams.tacotron_test_batches == self.test_steps

		#pad input sequences with the <pad_token> 0 ( _ )
		self._pad = 0
		#explicitely setting the padding to a value that doesn't originally exist in the spectogram
		#to avoid any possible conflicts, without affecting the output range of the model too much
		if hparams.symmetric_mels:
			self._target_pad = -hparams.max_abs_value
		else:
			self._target_pad = 0.
		#Mark finished sequences with 1s
		self._token_pad = 1.

		with tf.device('/cpu:0'):
			# Create placeholders for inputs and targets. Don't specify batch size because we want
			# to be able to feed different batch sizes at eval time.
			self._placeholders = [
			tf.placeholder(tf.int32, shape=(None, None), name='inputs'),
			tf.placeholder(tf.int32, shape=(None, ), name='input_lengths'),
			tf.placeholder(tf.float32, shape=(None, None, hparams.num_mels), name='mel_targets'),
			tf.placeholder(tf.float32, shape=(None, None), name='token_targets'),
			# tf.placeholder(tf.float32, shape=(None, None, hparams.num_freq), name='linear_targets'),
			tf.placeholder(tf.int32, shape=(None, ), name='targets_lengths'),
			tf.placeholder(tf.int32, shape=(hparams.tacotron_num_gpus, None), name='split_infos'),
      tf.placeholder(tf.int32, shape=(None,), name='emt_labels'),
      tf.placeholder(tf.int32, shape=(None,), name='spk_labels'),
			# tf.placeholder(tf.float32, shape=(None, hparams.tacotron_num_gpus*hparams.tacotron_spk_emb_dim), name='spk_emb'),
			tf.placeholder(tf.float32, shape=(None, None, hparams.num_mels), name='ref_mel_emt'),
			tf.placeholder(tf.float32, shape=(None, None, hparams.num_mels), name='ref_mel_spk'),
			tf.placeholder(tf.int32, shape=(None,), name='emt_up_labels'),
			tf.placeholder(tf.int32, shape=(None,), name='spk_up_labels'),
			tf.placeholder(tf.float32, shape=(None, None, hparams.num_mels), name='ref_mel_up_emt'),
			tf.placeholder(tf.float32, shape=(None, None, hparams.num_mels), name='ref_mel_up_spk')
			]

			# Create queue for buffering data
			queue_no = 1 if args.TEST else 8
			queue = tf.FIFOQueue(queue_no, [tf.int32, tf.int32, tf.float32, tf.float32,
															 #tf.float32,
															 tf.int32, tf.int32, tf.int32, tf.int32,
															 #tf.float32,
															 tf.float32, tf.float32,
															 tf.int32, tf.int32,
															 tf.float32, tf.float32], name='input_queue')
			self._enqueue_op = queue.enqueue(self._placeholders)
			# self.inputs, self.input_lengths, self.mel_targets, self.token_targets, self.linear_targets, self.targets_lengths,\
			# 	self.split_infos, self.emt_labels, self.spk_labels, self.spk_emb, self.ref_mel_emt, self.ref_mel_spk,\
			# 	self.ref_mel_up_emt, self.ref_mel_up_spk= queue.dequeue()
			self.inputs, self.input_lengths, self.mel_targets, self.token_targets, self.targets_lengths,\
				self.split_infos, self.emt_labels, self.spk_labels, self.ref_mel_emt, self.ref_mel_spk,self.emt_up_labels, self.spk_up_labels,\
				self.ref_mel_up_emt, self.ref_mel_up_spk= queue.dequeue()

			self.inputs.set_shape(self._placeholders[0].shape)
			self.input_lengths.set_shape(self._placeholders[1].shape)
			self.mel_targets.set_shape(self._placeholders[2].shape)
			self.token_targets.set_shape(self._placeholders[3].shape)
			#self.linear_targets.set_shape(self._placeholders[4].shape)
			self.targets_lengths.set_shape(self._placeholders[4].shape)
			self.split_infos.set_shape(self._placeholders[5].shape)
			self.emt_labels.set_shape(self._placeholders[6].shape)
			self.spk_labels.set_shape(self._placeholders[7].shape)

			#self.spk_emb.set_shape(self._placeholders[9].shape)
			self.ref_mel_emt.set_shape(self._placeholders[8].shape)
			self.ref_mel_spk.set_shape(self._placeholders[9].shape)
			self.emt_up_labels.set_shape(self._placeholders[10].shape)
			self.spk_up_labels.set_shape(self._placeholders[11].shape)
			self.ref_mel_up_emt.set_shape(self._placeholders[12].shape)
			self.ref_mel_up_spk.set_shape(self._placeholders[13].shape)

			# Create eval queue for buffering eval data
			eval_queue = tf.FIFOQueue(1, [tf.int32, tf.int32, tf.float32, tf.float32,
															 # tf.float32,
															 tf.int32, tf.int32, tf.int32, tf.int32,
																#tf.float32,
															 tf.float32, tf.float32], name='eval_queue')

			self._eval_enqueue_op = eval_queue.enqueue(self._placeholders[0:10])
			# self.eval_inputs, self.eval_input_lengths, self.eval_mel_targets, self.eval_token_targets, \
			# 	self.eval_linear_targets, self.eval_targets_lengths, self.eval_split_infos, self.eval_emt_labels,\
			# 	self.eval_spk_labels, self.eval_spk_emb, self.eval_ref_mel_emt, self.eval_ref_mel_spk = eval_queue.dequeue()
			self.eval_inputs, self.eval_input_lengths, self.eval_mel_targets, self.eval_token_targets, \
				self.eval_targets_lengths, self.eval_split_infos, self.eval_emt_labels,\
				self.eval_spk_labels, self.eval_ref_mel_emt, self.eval_ref_mel_spk = eval_queue.dequeue()

			self.eval_inputs.set_shape(self._placeholders[0].shape)
			self.eval_input_lengths.set_shape(self._placeholders[1].shape)
			self.eval_mel_targets.set_shape(self._placeholders[2].shape)
			self.eval_token_targets.set_shape(self._placeholders[3].shape)
			#self.eval_linear_targets.set_shape(self._placeholders[4].shape)
			self.eval_targets_lengths.set_shape(self._placeholders[4].shape)
			self.eval_split_infos.set_shape(self._placeholders[5].shape)
			self.eval_emt_labels.set_shape(self._placeholders[6].shape)
			self.eval_spk_labels.set_shape(self._placeholders[7].shape)
			#self.eval_spk_emb.set_shape(self._placeholders[9].shape)
			self.eval_ref_mel_emt.set_shape(self._placeholders[8].shape)
			self.eval_ref_mel_spk.set_shape(self._placeholders[9].shape)

	def start_threads(self, session):
		self._session = session
		thread = threading.Thread(name='background', target=self._enqueue_next_train_group)
		thread.daemon = True #Thread will close when parent quits
		thread.start()

		thread = threading.Thread(name='background', target=self._enqueue_next_test_group)
		thread.daemon = True #Thread will close when parent quits
		thread.start()

	def _get_test_groups(self):
		meta = self._test_meta[self._test_offset]
		self._test_offset += 1

		dataset = meta[0]
		text = meta[7]
		emt_label = meta[8]
		spk_label = meta[9]

		input_data = np.asarray(text_to_sequence(text, self._cleaner_names), dtype=np.int32)
		mel_target = np.load(os.path.join(self.data_folder,dataset,'mels', meta[2]))
		#Create parallel sequences containing zeros to represent a non finished sequence
		token_target = np.asarray([0.] * (len(mel_target) - 1))

		# linear_target_path = os.path.join(self.data_folder, dataset, 'linear', meta[3])
		# if hparams.predict_linear:
		# 	if os.path.exists(linear_target_path):
		# 		linear_target = np.load(linear_target_path)
		# 	else:
		# 		raise ("linear target does not exist -", linear_target_path)
		# else:
		# 	linear_target = np.zeros((1,hparams.num_freq))

		#check for speaker embedding
		# spk_emb_path = os.path.join(self.data_folder,dataset,'spkemb', meta[4])
		# if os.path.exists(spk_emb_path):
		# 	spk_emb = np.load(spk_emb_path)
		# else:
		# 	spk_emb = np.zeros(hparams.tacotron_spk_emb_dim)
		# assert spk_emb.shape[0] == hparams.tacotron_spk_emb_dim

		#just use the same sample for the reference when testing
		ref_mel_emt = mel_target
		ref_mel_spk = mel_target

		# return (input_data, mel_target, token_target, linear_target, spk_emb, emt_label, spk_label, ref_mel_emt, ref_mel_spk, len(mel_target))
		return (input_data, mel_target, token_target, emt_label, spk_label, ref_mel_emt, ref_mel_spk, len(mel_target))

	def make_test_batches(self):
		start = time.time()

		# Read a group of examples
		n = self._hparams.tacotron_batch_size
		r = self._hparams.outputs_per_step

		#Test on entire test set
		test_batches_per_group = n*2 if self._args.test_max_len or self._args.TEST else len(self._test_meta)

		examples = [self._get_test_groups() for i in range(test_batches_per_group)]

		# Bucket examples based on similar output sequence length for efficiency
		examples.sort(key=lambda x: x[-1])
		batches = [examples[i: i+n] for i in range(0, len(examples), n)]

		if self._args.test_max_len:
			batches = batches[::-1]
		else:
			np.random.shuffle(batches)

		log('\nGenerated {} test batches of size {} in {:.3f} sec'.format(len(batches), n, time.time() - start))
		return batches, r

	def _enqueue_next_train_group(self):
		while not self._coord.should_stop():
			start = time.time()

			# Read a group of examples
			n = self._hparams.tacotron_batch_size
			r = self._hparams.outputs_per_step
			examples = [self._get_next_example() for i in range(n * self._batches_per_group)]

			# Bucket examples based on similar output sequence length for efficiency
			examples.sort(key=lambda x: x[-1])
			batches = [examples[i: i+n] for i in range(0, len(examples), n)]
			if self._args.test_max_len:
				batches = batches[::-1]
			else:
				np.random.shuffle(batches)

			log('\nGenerated {} train batches of size {} in {:.3f} sec'.format(len(batches), n, time.time() - start))
			for batch in batches:
				feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch, r, TRAIN=True)))
				self._session.run(self._enqueue_op, feed_dict=feed_dict)

	def _enqueue_next_test_group(self):
		#Create test batches once and evaluate on them for all test steps
		test_batches, r = self.make_test_batches()
		while not self._coord.should_stop():
			for batch in test_batches:
				feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch, r, TRAIN=False)))
				self._session.run(self._eval_enqueue_op, feed_dict=feed_dict)

	def _get_next_example(self):
		"""Gets a single example (input, mel_target, token_target, linear_target, mel_length) from_ disk
		"""
		if self._train_offset >= len(self._train_meta):
			self._train_offset = 0
			np.random.shuffle(self._train_meta)

		meta = self._train_meta[self._train_offset]
		self._train_offset += 1

		dataset = meta[0]
		text = meta[7]
		emt_label = meta[8]
		spk_label = meta[9]

		input_data = np.asarray(text_to_sequence(text, self._cleaner_names), dtype=np.int32)
		mel_target = np.load(os.path.join(self.data_folder, dataset, 'mels', meta[2]))
		#Create parallel sequences containing zeros to represent a non finished sequence
		token_target = np.asarray([0.] * (len(mel_target) - 1))

		# linear_target_path = os.path.join(self.data_folder, dataset, 'linear', meta[3])
		# if hparams.predict_linear:
		# 	if os.path.exists(linear_target_path):
		# 		linear_target = np.load(linear_target_path)
		# 	else:
		# 		raise ("linear target does not exist -", linear_target_path)
		# else:
		# 	linear_target = np.zeros((1,hparams.num_freq))

		#check for speaker embedding
		# spk_emb_path = os.path.join(self.data_folder,dataset,'spkemb', meta[4])
		# if os.path.exists(spk_emb_path):
		# 	spk_emb = np.load(spk_emb_path)
		# else:
		# 	spk_emb = np.zeros(hparams.tacotron_spk_emb_dim)
		# assert spk_emb.shape[0] == hparams.tacotron_spk_emb_dim

		ref_mel_up_emt = np.zeros((1,hparams.num_mels))
		ref_mel_up_spk = np.zeros((1,hparams.num_mels))
		emt_up_label=emt_label.copy()
		spk_up_label=spk_label.copy()

		if self._args.intercross_both:
			chosen_cross = np.random.choice(['emt','spk'])
			label = emt_label if chosen_cross == 'emt' else spk_label
			column_name = 'emt_label' if chosen_cross == 'emt' else 'spk_label'
			df_meta_same_style = self.df_meta_train[self.df_meta_train.loc[:, column_name] == label]
			# select one mel from same style to use as reference
			idx = np.random.choice(df_meta_same_style.index)
			mel_name_same_style = df_meta_same_style.loc[idx, 'mel_filename']
			dataset_same_style = df_meta_same_style.loc[idx, 'dataset']
			ref_mel_same_style = np.load(os.path.join(self.data_folder, dataset_same_style, 'mels', mel_name_same_style))

			ref_mel_emt = ref_mel_same_style if chosen_cross=='emt' else mel_target
			ref_mel_spk = mel_target if chosen_cross == 'emt' else ref_mel_same_style

		else:
			if dataset == 'emt4' or dataset == 'emth': #np.random.choice(['emt','spk']) == 'emt':
				ref_mel_spk = mel_target
				# find all mels with same emotion type
				df_meta_same_style = self.df_meta_train[self.df_meta_train['dataset'].isin(['emt4','emth'])]
				df_meta_same_style = df_meta_same_style[df_meta_same_style['emt_label'] == emt_label]

				# select one mel from same style to use as reference
				idx = np.random.choice(df_meta_same_style.index)
				mel_name_same_style = df_meta_same_style.loc[idx, 'mel_filename']
				dataset_same_style = df_meta_same_style.loc[idx, 'dataset']
				ref_mel_emt = np.load(os.path.join(self.data_folder, dataset_same_style, 'mels', mel_name_same_style))

			elif dataset == 'librispeech' or 'vctk':
				ref_mel_emt = mel_target
				# find all mels with same spk type
				df_meta_same_style = self.df_meta_train[self.df_meta_train.loc[:, 'spk_label'] == spk_label]

				# select one mel from same style to use as reference
				idx = np.random.choice(df_meta_same_style.index)
				mel_name_same_style = df_meta_same_style.loc[idx, 'mel_filename']
				dataset_same_style = df_meta_same_style.loc[idx, 'dataset']
				ref_mel_spk = np.load(os.path.join(self.data_folder,dataset_same_style,'mels', mel_name_same_style))
			else:
				raise ValueError('Invalid dataset type')

		if self._args.unpaired:
			# chose a random emotion and a random spk id (this keeps it even amongst all classes)
			emt_up_label = np.random.choice(self.total_emt)
			spk_up_label = np.random.choice(self.total_spk)
			df_meta_same_emt = self.df_meta_train[self.df_meta_train.loc[:, 'emt_label'] == emt_up_label]
			df_meta_same_spk = self.df_meta_train[self.df_meta_train.loc[:, 'spk_label'] == spk_up_label]
			for i, df in enumerate([df_meta_same_emt,df_meta_same_spk]):
				idx = np.random.choice(df.index)
				mel_name_up = df.loc[idx, 'mel_filename']
				dataset_up = df.loc[idx, 'dataset']
				mel = np.load(os.path.join(self.data_folder, dataset_up, 'mels', mel_name_up))
				if i == 0:
					ref_mel_up_emt = mel.copy()
				else:
					ref_mel_up_spk = mel.copy()

		# return (input_data, mel_target, token_target, linear_target, spk_emb, emt_label, spk_label, ref_mel_emt, ref_mel_spk, len(mel_target))
		return (input_data, mel_target, token_target, emt_label, spk_label, ref_mel_emt, ref_mel_spk, emt_up_label, spk_up_label,
						ref_mel_up_emt, ref_mel_up_spk, len(mel_target))

	def _prepare_batch(self, batches, outputs_per_step, TRAIN):
		assert 0 == len(batches) % self._hparams.tacotron_num_gpus
		size_per_device = int(len(batches) / self._hparams.tacotron_num_gpus)
		np.random.shuffle(batches)

		inputs = None
		mel_targets = None
		token_targets = None
		linear_targets = None
		spk_embs = None
		split_infos = []
		mel_refs_emt = None
		mel_refs_spk = None
		mel_refs_up_emt = None
		mel_refs_up_spk = None
		linear_target_max_len = 0
		mel_refs_up_emt_max_len = 0
		mel_refs_up_spk_max_len = 0

		IDX_INP = 0
		IDX_MEL = 1
		IDX_TOK = 2
		# IDX_LIN = 3
		# IDX_SPK_EMB = 4
		IDX_EMT_LBL = 3
		IDX_SPK_LBL = 4
		IDX_REF_EMT = 5
		IDX_REF_SPK = 6
		IDX_EMT_UP_LBL = 7
		IDX_SPK_UP_LBL = 8
		IDX_REF_UP_EMT = 9
		IDX_REF_UP_SPK = 10
		IDX_LEN = -1

		targets_lengths = np.asarray([x[IDX_LEN] for x in batches], dtype=np.int32) #Used to mask loss
		emt_labels = np.asarray([x[IDX_EMT_LBL] for x in batches], dtype = np.int32)
		spk_labels = np.asarray([x[IDX_SPK_LBL] for x in batches], dtype=np.int32)
		if TRAIN:
			emt_up_labels = np.asarray([x[IDX_EMT_UP_LBL] for x in batches], dtype = np.int32)
			spk_up_labels = np.asarray([x[IDX_SPK_UP_LBL] for x in batches], dtype=np.int32)
		input_lengths = np.asarray([len(x[IDX_INP]) for x in batches], dtype=np.int32)

		#Produce inputs/targets of variables lengths for different GPUs
		for i in range(self._hparams.tacotron_num_gpus):
			batch = batches[size_per_device * i: size_per_device * (i + 1)]

			input_cur_device, input_max_len = self._prepare_inputs([x[IDX_INP] for x in batch])
			inputs = np.concatenate((inputs, input_cur_device), axis=1) if inputs is not None else input_cur_device
			mel_target_cur_device, mel_target_max_len = self._prepare_targets([x[IDX_MEL] for x in batch], outputs_per_step)
			mel_targets = np.concatenate(( mel_targets, mel_target_cur_device), axis=1) if mel_targets is not None else mel_target_cur_device

			#Pad sequences with 1 to infer that the sequence is done
			token_target_cur_device, token_target_max_len = self._prepare_token_targets([x[IDX_TOK] for x in batch], outputs_per_step)
			token_targets = np.concatenate((token_targets, token_target_cur_device),axis=1) if token_targets is not None else token_target_cur_device
			# linear_targets_cur_device, linear_target_max_len = self._prepare_targets([x[IDX_LIN] for x in batch], outputs_per_step)
			# linear_targets = np.concatenate((linear_targets, linear_targets_cur_device), axis=1) if linear_targets is not None else linear_targets_cur_device

			# spk_emb_cur_device = np.stack([x[IDX_SPK_EMB] for x in batch])
			# spk_embs = np.concatenate((spk_embs, spk_emb_cur_device), axis=1) if spk_embs is not None else spk_emb_cur_device

			mel_refs_emt_cur_device, mel_refs_emt_max_len = self._prepare_targets([x[IDX_REF_EMT] for x in batch], outputs_per_step)
			mel_refs_emt = np.concatenate(( mel_refs_emt, mel_refs_emt_cur_device), axis=1) if mel_refs_emt is not None else mel_refs_emt_cur_device

			mel_refs_spk_cur_device, mel_refs_spk_max_len = self._prepare_targets([x[IDX_REF_SPK] for x in batch], outputs_per_step)
			mel_refs_spk = np.concatenate(( mel_refs_spk, mel_refs_spk_cur_device), axis=1) if mel_refs_spk is not None else mel_refs_spk_cur_device

			if TRAIN:
				mel_refs_up_emt_cur_device, mel_refs_up_emt_max_len = self._prepare_targets([x[IDX_REF_UP_EMT] for x in batch],outputs_per_step)
				mel_refs_up_emt = np.concatenate((mel_refs_up_emt, mel_refs_up_emt_cur_device),axis=1) if mel_refs_up_emt is not None else mel_refs_up_emt_cur_device

				mel_refs_up_spk_cur_device, mel_refs_up_spk_max_len = self._prepare_targets([x[IDX_REF_UP_SPK] for x in batch],outputs_per_step)
				mel_refs_up_spk = np.concatenate((mel_refs_up_spk, mel_refs_up_spk_cur_device),axis=1) if mel_refs_up_spk is not None else mel_refs_up_spk_cur_device


			split_infos.append([input_max_len, mel_target_max_len, token_target_max_len, linear_target_max_len,
													hparams.tacotron_spk_emb_dim, mel_refs_emt_max_len, mel_refs_spk_max_len,
													mel_refs_up_emt_max_len,mel_refs_up_spk_max_len])

		split_infos = np.asarray(split_infos, dtype=np.int32)

		# return (inputs, input_lengths, mel_targets, token_targets, linear_targets, targets_lengths, split_infos, emt_labels,
		# 			spk_labels, spk_embs, mel_refs_emt, mel_refs_spk)
		if TRAIN:
			return (inputs, input_lengths, mel_targets, token_targets, targets_lengths, split_infos, emt_labels, spk_labels,
							mel_refs_emt, mel_refs_spk, emt_up_labels, spk_up_labels, mel_refs_up_emt, mel_refs_up_spk)
		else:
			return (inputs, input_lengths, mel_targets, token_targets, targets_lengths, split_infos, emt_labels, spk_labels, mel_refs_emt, mel_refs_spk)

	def _prepare_inputs(self, inputs):
		max_len = max([len(x) for x in inputs])
		return np.stack([self._pad_input(x, max_len) for x in inputs]), max_len

	def _prepare_targets(self, targets, alignment):
		max_len = max([len(t) for t in targets])
		data_len = self._round_up(max_len, alignment)
		return np.stack([self._pad_target(t, data_len) for t in targets]), data_len

	def _prepare_token_targets(self, targets, alignment):
		max_len = max([len(t) for t in targets]) + 1
		data_len = self._round_up(max_len, alignment)
		return np.stack([self._pad_token_target(t, data_len) for t in targets]), data_len

	def _pad_input(self, x, length):
		return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=self._pad)

	def _pad_target(self, t, length):
		return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=self._target_pad)

	def _pad_token_target(self, t, length):
		return np.pad(t, (0, length - t.shape[0]), mode='constant', constant_values=self._token_pad)

	def _round_up(self, x, multiple):
		remainder = x % multiple
		return x if remainder == 0 else x + multiple - remainder

	def _round_down(self, x, multiple):
		remainder = x % multiple
		return x if remainder == 0 else x - remainder

def test():

	parser = argparse.ArgumentParser()
	parser.add_argument('--intercross', action='store_true', default=False, help='whether to use intercross training')
	args = parser.parse_args()

	args.tacotron_test_steps = 3
	args.remove_long_samps = False#True
	args.test_max_len = False#True
	args.unpaired = True#False#True
	args.TEST = True
	args.intercross_both = False

	metadata_filename = 'C:/Users/t-mawhit/Documents/code/Tacotron-2/data/train_emt4_vctk_emth.txt'
	coord = tf.train.Coordinator()
	feeder = Feeder(coord, metadata_filename, hparams, args)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		feeder.start_threads(sess)
		# vars = [feeder.inputs, feeder.input_lengths, feeder.mel_targets, feeder.token_targets, feeder.linear_targets,
		# 				feeder.targets_lengths, feeder.split_infos, feeder.emt_labels, feeder.spk_labels, feeder.spk_emb,
		# 				feeder.ref_mel_emt, feeder.ref_mel_spk]
		vars = [feeder.inputs, feeder.input_lengths, feeder.mel_targets, feeder.token_targets,
						feeder.targets_lengths, feeder.split_infos, feeder.emt_labels, feeder.spk_labels, feeder.ref_mel_emt, feeder.ref_mel_spk]
		if args.unpaired:
			vars += [feeder.emt_up_labels,feeder.spk_up_labels,feeder.ref_mel_up_emt, feeder.ref_mel_up_spk]

		outputs = sess.run(vars)

		if args.unpaired:
			(inputs, input_lengths, mel_targets, token_targets, targets_lengths, split_infos, emt_labels, spk_labels,
			 ref_mel_emt, ref_mel_spk,emt_up_labels, spk_up_labels, ref_mel_up_emt,ref_mel_up_spk) = outputs
		else:
			(inputs, input_lengths, mel_targets, token_targets, targets_lengths, split_infos, emt_labels, spk_labels, ref_mel_emt,
			 ref_mel_spk) = outputs

		print("mel_targets", len(mel_targets), mel_targets[0].shape)
		# print("linear_targets", len(linear_targets), linear_targets[0].shape)
		print("inputs", inputs.shape)
		print("token_targets", token_targets.shape)
		# print("spk_emb", spk_emb.shape)
		print("input_lengths", input_lengths.shape)
		print("targets_lengths", input_lengths.shape)
		print("emt_labels", emt_labels.shape)
		print("spk_labels", spk_labels.shape)
		print("split_infos", split_infos.shape)
		print("ref_mel_emt", ref_mel_emt.shape)
		print("ref_mel_spk", ref_mel_spk.shape)
		if args.unpaired:
			print("emt_up_labels", emt_up_labels.shape)
			print("spk_up_labels", spk_up_labels.shape)
			print("ref_mel_up_emt", ref_mel_up_emt.shape)
			print("ref_mel_up_spk", ref_mel_up_spk.shape)

if __name__ == '__main__':
	test()