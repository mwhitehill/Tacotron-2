import argparse
import os
import subprocess
import time
import traceback
from datetime import datetime

import infolog
import numpy as np
import tensorflow as tf
from datasets import audio
from hparams import hparams_debug_string
from tacotron.feeder import Feeder
from tacotron.models import create_model
from tacotron.utils import ValueWindow, plot
from tacotron.utils.text import sequence_to_text
from tacotron.utils.symbols import symbols
from tacotron.synthesize import get_filenames_from_metadata
from tacotron.synthesizer import filenames_to_inputs, get_output_lengths
from tqdm import tqdm

log = infolog.log


def time_string():
	return datetime.now().strftime('%Y-%m-%d %H:%M')

def get_eval_feed_dict(hparams, synth_metadata_filename, eval_model, input_dir, flip_spk_emt):

	#eval synthesis data
	texts, basenames, basenames_refs, mel_filenames, \
	mel_ref_filenames_emt, mel_ref_filenames_spk, \
	emt_labels, spk_labels = get_filenames_from_metadata(synth_metadata_filename,
														 input_dir, flip_spk_emt)

	basenames, basenames_refs, inputs, input_lengths, split_infos, mel_refs_emt, mel_refs_spk, \
	emt_labels, spk_labels = filenames_to_inputs(hparams, texts, basenames, mel_filenames,
												 basenames_refs, mel_ref_filenames_emt,
												 mel_ref_filenames_spk, emt_labels,
												 spk_labels)
	feed_dict = {
		eval_model.synth_inputs: inputs,
		eval_model.synth_input_lengths: input_lengths,
		eval_model.synth_split_infos: split_infos,
		eval_model.synth_mel_refs_emt: mel_refs_emt,
		eval_model.synth_mel_refs_spk: mel_refs_spk,
	}

	return(feed_dict, emt_labels, spk_labels,basenames, basenames_refs)


def add_embedding_stats(summary_writer, embedding_names, paths_to_meta, checkpoint_path):
	#Create tensorboard projector
	config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
	config.model_checkpoint_path = checkpoint_path

	for embedding_name, path_to_meta in zip(embedding_names, paths_to_meta):
		#Initialize config
		embedding = config.embeddings.add()
		#Specifiy the embedding variable and the metadata
		embedding.tensor_name = embedding_name
		embedding.metadata_path = path_to_meta
	
	#Project the embeddings to space dimensions for visualization
	tf.contrib.tensorboard.plugins.projector.visualize_embeddings(summary_writer, config)

def add_train_stats(model, hparams):
	with tf.variable_scope('stats') as scope:
		for i in range(hparams.tacotron_num_gpus):
			tf.summary.histogram('mel_outputs %d' % i, model.tower_mel_outputs[i])
			tf.summary.histogram('mel_targets %d' % i, model.tower_mel_targets[i])
		tf.summary.scalar('before_loss', model.before_loss)
		tf.summary.scalar('after_loss', model.after_loss)

		if hparams.predict_linear:
			tf.summary.scalar('linear_loss', model.linear_loss)
			for i in range(hparams.tacotron_num_gpus):
				tf.summary.histogram('linear_outputs %d' % i, model.tower_linear_outputs[i])
				tf.summary.histogram('linear_targets %d' % i, model.tower_linear_targets[i])
		
		tf.summary.scalar('regularization_loss', model.regularization_loss)
		tf.summary.scalar('stop_token_loss', model.stop_token_loss)
		tf.summary.scalar('loss', model.loss)
		tf.summary.scalar('learning_rate', model.learning_rate) #Control learning rate decay speed
		if hparams.tacotron_teacher_forcing_mode == 'scheduled':
			tf.summary.scalar('teacher_forcing_ratio', model.ratio) #Control teacher forcing ratio decay when mode = 'scheduled'
		gradient_norms = [tf.norm(grad) for grad in model.gradients]
		tf.summary.histogram('gradient_norm', gradient_norms)
		tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms)) #visualize gradients (in case of explosion)
		return tf.summary.merge_all()

def add_eval_stats(summary_writer, step, linear_loss, before_loss, after_loss, stop_token_loss, loss):
	values = [
	tf.Summary.Value(tag='Tacotron_eval_model/eval_stats/eval_before_loss', simple_value=before_loss),
	tf.Summary.Value(tag='Tacotron_eval_model/eval_stats/eval_after_loss', simple_value=after_loss),
	tf.Summary.Value(tag='Tacotron_eval_model/eval_stats/stop_token_loss', simple_value=stop_token_loss),
	tf.Summary.Value(tag='Tacotron_eval_model/eval_stats/eval_loss', simple_value=loss),
	]
	if linear_loss is not None:
		values.append(tf.Summary.Value(tag='Tacotron_eval_model/eval_stats/eval_linear_loss', simple_value=linear_loss))
	test_summary = tf.Summary(value=values)
	summary_writer.add_summary(test_summary, step)

def model_train_mode(args, feeder, hparams, global_step):
	with tf.variable_scope('Tacotron_model', reuse=tf.AUTO_REUSE) as scope:
		model_name='Tacotron_emt_attn' if args.emt_attn else 'Tacotron'
		model = create_model(model_name, hparams)
		if hparams.predict_linear:
			raise ValueError('predict linear not implemented')
			model.initialize(args, feeder.inputs, feeder.input_lengths, feeder.mel_targets, feeder.token_targets, linear_targets=feeder.linear_targets,
				targets_lengths=feeder.targets_lengths, global_step=global_step,
				is_training=True, split_infos=feeder.split_infos, emt_labels = feeder.emt_labels, spk_labels = feeder.spk_labels,
				emt_up_labels = feeder.emt_up_labels, spk_up_labels = feeder.spk_up_labels, spk_emb= feeder.spk_emb,
				ref_mel_emt=feeder.ref_mel_emt, ref_mel_spk= feeder.ref_mel_spk, use_emt_disc = args.emt_disc, use_spk_disc = args.spk_disc,
				use_intercross=args.intercross, n_emt = len(feeder.total_emt), n_spk = len(feeder.total_spk))
		else:
			emt_up_labels = feeder.emt_up_labels if args.unpaired else None
			spk_up_labels = feeder.spk_up_labels if args.unpaired else None
			ref_mel_up_emt = feeder.ref_mel_up_emt if args.unpaired else None
			ref_mel_up_spk = feeder.ref_mel_up_spk if args.unpaired else None

			ref_mel_emt = feeder.ref_mel_emt if not(args.flip_spk_emt) else feeder.ref_mel_spk
			ref_mel_spk = feeder.ref_mel_spk if not (args.flip_spk_emt) else feeder.ref_mel_emt
			emt_labels = feeder.emt_labels if not (args.flip_spk_emt) else feeder.spk_labels
			spk_labels = feeder.spk_labels if not (args.flip_spk_emt) else feeder.emt_labels
			n_emt = len(feeder.total_emt) if not (args.flip_spk_emt) else len(feeder.total_spk)
			n_spk = len(feeder.total_spk) if not (args.flip_spk_emt) else len(feeder.total_emt)

			model.initialize(args, feeder.inputs, feeder.input_lengths, feeder.mel_targets, feeder.token_targets,
				targets_lengths=feeder.targets_lengths, global_step=global_step,
				is_training=True, split_infos=feeder.split_infos, emt_labels = emt_labels, spk_labels = spk_labels,
				emt_up_labels = emt_up_labels, spk_up_labels = spk_up_labels, ref_mel_emt=ref_mel_emt, ref_mel_spk=ref_mel_spk,
				ref_mel_up_emt=ref_mel_up_emt, ref_mel_up_spk=ref_mel_up_spk, use_emt_disc = args.emt_disc,
				use_spk_disc = args.spk_disc, use_intercross=args.intercross, use_unpaired=args.unpaired,
				n_emt=n_emt, n_spk=n_spk)
		model.add_loss()
		model.add_optimizer(global_step)
		stats = add_train_stats(model, hparams)
		return model, stats

def model_test_mode(args, hparams, train_model): #feeder,global_step):
	with tf.variable_scope('Tacotron_model', reuse=tf.AUTO_REUSE) as scope:

		model_name = 'Tacotron_emt_attn' if args.emt_attn else 'Tacotron'
		model = create_model(model_name, hparams)

		model.synth_inputs = tf.placeholder(tf.int32, (None, None), name='synth_inputs')
		model.synth_input_lengths = tf.placeholder(tf.int32, (None), name='synth_input_lengths')
		model.synth_split_infos = tf.placeholder(tf.int32, shape=(hparams.tacotron_num_gpus, None), name='synth_split_infos')

		model.synth_mel_refs_emt = tf.placeholder(tf.float32, (None, None, hparams.num_mels), name='synth_mel_refs_emt')
		model.synth_mel_refs_spk = tf.placeholder(tf.float32, (None, None, hparams.num_mels), name='synth_mel_refs_spk')



		model.initialize(args, model.synth_inputs, model.synth_input_lengths, split_infos=model.synth_split_infos,
						 ref_mel_emt=model.synth_mel_refs_emt, ref_mel_spk=model.synth_mel_refs_spk,
						 n_emt=train_model.n_emt, n_spk=train_model.n_spk)

		# model.add_loss()

		# ref_mel_emt = feeder.eval_ref_mel_emt if not(args.flip_spk_emt) else feeder.eval_ref_mel_spk
		# ref_mel_spk = feeder.eval_ref_mel_spk if not (args.flip_spk_emt) else feeder.eval_ref_mel_emt
		# emt_labels = feeder.eval_emt_labels if not (args.flip_spk_emt) else feeder.eval_spk_labels
		# spk_labels = feeder.eval_spk_labels if not (args.flip_spk_emt) else feeder.eval_emt_labels
		# n_emt = len(feeder.total_emt) if not (args.flip_spk_emt) else len(feeder.total_spk)
		# n_spk = len(feeder.total_spk) if not (args.flip_spk_emt) else len(feeder.total_emt)

		# model.initialize(args, feeder.eval_inputs, feeder.eval_input_lengths, feeder.eval_mel_targets, feeder.eval_token_targets,
		# 	targets_lengths=feeder.eval_targets_lengths, global_step=global_step, is_training=False, is_evaluating=True,
		# 	split_infos=feeder.eval_split_infos, emt_labels=emt_labels, spk_labels=spk_labels,
		# 	ref_mel_emt=ref_mel_emt, ref_mel_spk= ref_mel_spk, use_emt_disc = args.emt_disc, use_spk_disc = args.spk_disc,
		# 	use_intercross=args.intercross, n_emt = n_emt, n_spk =n_spk)

		return model

def train(log_dir, args, hparams):
	save_dir = os.path.join(log_dir, 'taco_pretrained')
	plot_dir = os.path.join(log_dir, 'plots')
	wav_dir = os.path.join(log_dir, 'wavs')
	mel_dir = os.path.join(log_dir, 'mel-spectrograms')
	eval_dir = os.path.join(log_dir, 'eval-dir')
	eval_plot_dir = os.path.join(eval_dir, 'plots')
	eval_wav_dir = os.path.join(eval_dir, 'wavs')
	tensorboard_dir = os.path.join(log_dir, 'tacotron_events')
	meta_folder = os.path.join(log_dir, 'metas')
	os.makedirs(save_dir, exist_ok=True)
	os.makedirs(plot_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)
	os.makedirs(mel_dir, exist_ok=True)
	os.makedirs(eval_dir, exist_ok=True)
	os.makedirs(eval_plot_dir, exist_ok=True)
	os.makedirs(eval_wav_dir, exist_ok=True)
	os.makedirs(tensorboard_dir, exist_ok=True)
	os.makedirs(meta_folder, exist_ok=True)

	checkpoint_path = os.path.join(save_dir, 'tacotron_model.ckpt')
	input_path = os.path.join(args.base_dir, args.tacotron_input)

	if hparams.predict_linear:
		linear_dir = os.path.join(log_dir, 'linear-spectrograms')
		os.makedirs(linear_dir, exist_ok=True)

	log('Checkpoint path: {}'.format(checkpoint_path))
	log('Loading training data from: {}'.format(input_path))
	log('Using model: {}'.format(args.model))
	log(hparams_debug_string())

	#Start by setting a seed for repeatability
	tf.set_random_seed(hparams.tacotron_random_seed)

	#Set up data feeder
	coord = tf.train.Coordinator()
	with tf.variable_scope('datafeeder') as scope:
		feeder = Feeder(coord, input_path, hparams, args)

	#Set up model:
	global_step = tf.Variable(0, name='global_step', trainable=False)
	model, stats = model_train_mode(args, feeder, hparams, global_step)
	eval_model = model_test_mode(args, hparams, model)
	# if args.TEST:
	# 	for v in tf.global_variables():
	# 		print(v)

	#Embeddings metadata
	char_embedding_meta = os.path.join(meta_folder, 'CharacterEmbeddings.tsv')
	if not os.path.isfile(char_embedding_meta):
		with open(char_embedding_meta, 'w', encoding='utf-8') as f:
			for symbol in symbols:
				if symbol == ' ':
					symbol = '\\s' #For visual purposes, swap space with \s

				f.write('{}\n'.format(symbol))

	char_embedding_meta = char_embedding_meta.replace(log_dir, '..')

	#Potential Griffin-Lim GPU setup
	if hparams.GL_on_GPU:
		GLGPU_mel_inputs = tf.placeholder(tf.float32, (None, hparams.num_mels), name='GLGPU_mel_inputs')
		GLGPU_lin_inputs = tf.placeholder(tf.float32, (None, hparams.num_freq), name='GLGPU_lin_inputs')

		GLGPU_mel_outputs = audio.inv_mel_spectrogram_tensorflow(GLGPU_mel_inputs, hparams)
		GLGPU_lin_outputs = audio.inv_linear_spectrogram_tensorflow(GLGPU_lin_inputs, hparams)

	#Book keeping
	step = 0
	time_window = ValueWindow(100)
	loss_window = ValueWindow(100)
	loss_bef_window = ValueWindow(100)
	loss_aft_window = ValueWindow(100)
	loss_stop_window = ValueWindow(100)
	loss_reg_window = ValueWindow(100)
	loss_emt_window = ValueWindow(100)
	loss_spk_window = ValueWindow(100)
	loss_orthog_window = ValueWindow(100)
	loss_up_emt_window = ValueWindow(100)
	loss_up_spk_window = ValueWindow(100)
	loss_mo_up_emt_window = ValueWindow(100)
	loss_mo_up_spk_window = ValueWindow(100)
	if args.nat_gan:
		d_loss_t_window = ValueWindow(100)
		d_loss_p_window = ValueWindow(100)
		d_loss_up_window = ValueWindow(100)
		g_loss_p_window = ValueWindow(100)
		g_loss_up_window = ValueWindow(100)

	saver = tf.train.Saver(max_to_keep=args.max_to_keep)

	if args.opt_ref_no_mo and not(args.restart_optimizer_r):
		print("WILL ATTEMPT TO RESTORE OPTIMIZER R - SET ARGS.RESTART_OPTIMIZER_R IF RETRAINING A MODEL THAT DIDN'T HAVE THE OPTIMIZER R")

	assert(not(args.restart_nat_gan_d and args.restore_nat_gan_d_sep))

	var_list = tf.global_variables()
	var_list = [v for v in var_list if not ('pretrained' in v.name)]
	var_list = [v for v in var_list if not ('nat_gan' in v.name or 'optimizer_n' in v.name)] if (args.restart_nat_gan_d or args.restore_nat_gan_d_sep) else var_list
	var_list = [v for v in var_list if not ('optimizer_r' in v.name or 'optimizer_3' in v.name)] if args.restart_optimizer_r else var_list
	saver_restore = tf.train.Saver(var_list=var_list)

	if args.unpaired and args.pretrained_emb_disc:
		saver_restore_emt_disc = tf.train.Saver(var_list=[v for v in tf.global_variables() if ('pretrained_ref_enc_emt' in v.name)])
		saver_restore_spk_disc = tf.train.Saver(var_list=[v for v in tf.global_variables() if ('pretrained_ref_enc_spk' in v.name)])
	elif args.pretrained_emb_disc_all:
		saver_restore_emt_disc = tf.train.Saver(var_list=[v for v in tf.global_variables() if ('refnet_emt' in v.name)])
		saver_restore_spk_disc = tf.train.Saver(var_list=[v for v in tf.global_variables() if ('refnet_spk' in v.name)])

	if args.nat_gan:
		saver_nat_gan = tf.train.Saver(var_list=[v for v in tf.global_variables() if ('nat_gan' in v.name or 'optimizer_n' in v.name)])
		save_dir_nat_gan = r'nat_gan/pretrained_model'

	log('Tacotron training set to a maximum of {} steps'.format(args.tacotron_train_steps))
	if hparams.tacotron_fine_tuning:
		print('FINE TUNING SET TO TRUE - MAKE SURE THIS IS WHAT YOU WANT!')

	#Memory allocation on the GPU as needed
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True

	eval_feed_dict, emt_labels, spk_labels, \
	basenames, basenames_refs = get_eval_feed_dict(hparams, args.synth_metadata_filename,
												   eval_model, args.input_dir, args.flip_spk_emt)

	#Train
	with tf.Session(config=config) as sess:
		try:
			summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
			# for x in tf.global_variables():
			# 	print(x)

			sess.run(tf.global_variables_initializer())
			#saved model restoring
			if args.restore:
				# Restore saved model if the user requested it, default = True
				try:
					checkpoint_state = tf.train.get_checkpoint_state(save_dir)

					if (checkpoint_state and checkpoint_state.model_checkpoint_path):
						log('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path), slack=True)
						saver_restore.restore(sess, checkpoint_state.model_checkpoint_path)

					else:
						raise ValueError('No model to load at {}'.format(save_dir))

				except tf.errors.OutOfRangeError as e:
					log('Cannot restore checkpoint: {}'.format(e), slack=True)
			else:
				log('Starting new training!', slack=True)
				saver.save(sess, checkpoint_path, global_step=global_step)

			if args.unpaired and args.pretrained_emb_disc:
				save_dir_emt = r'spk_disc/pretrained_model_emt_disc'
				checkpoint_state_emt = tf.train.get_checkpoint_state(save_dir_emt)
				saver_restore_emt_disc.restore(sess, checkpoint_state_emt.model_checkpoint_path)
				log('Loaded Emotion Discriminator from checkpoint {}'.format(checkpoint_state_emt.model_checkpoint_path), slack=True)

				save_dir_spk = r'spk_disc/pretrained_model_spk_disc'
				checkpoint_state_spk = tf.train.get_checkpoint_state(save_dir_spk)
				saver_restore_spk_disc.restore(sess, checkpoint_state_spk.model_checkpoint_path)
				log('Loaded Speaker Discriminator from checkpoint {}'.format(checkpoint_state_spk.model_checkpoint_path), slack=True)

			if args.nat_gan and args.restore_nat_gan_d_sep:
				checkpoint_state_nat_gan = tf.train.get_checkpoint_state(save_dir_nat_gan)
				saver_nat_gan.restore(sess, checkpoint_state_nat_gan.model_checkpoint_path)
				log('Loaded Nat Gan Discriminator from checkpoint {}'.format(checkpoint_state_nat_gan.model_checkpoint_path), slack=True)

			#initializing feeder
			feeder.start_threads(sess)

			#Training loop
			while not coord.should_stop() and step < args.tacotron_train_steps:
				start_time = time.time()
				# vars = [global_step, model.loss, model.optimize,model.before_loss, model.after_loss,model.stop_token_loss,
				# 				model.regularization_loss,model.style_emb_loss_emt, model.style_emb_loss_spk, model.style_emb_orthog_loss]
				# out = [step, loss, opt, bef, aft, stop, reg, loss_emt, loss_spk, loss_orthog]
				# message = 'Step {:7d} {:.3f} sec/step, loss={:.5f}, avg_loss={:.5f}, bef={:.5f}, aft={:.5f}, stop={:.5f},' \
				# 					'reg={:.5f}, emt={:.5f}, spk={:.5f}, orthog={:.5f}'.format(step, time_window.average, loss, loss_window.average,
				# 																																		 loss_bef_window.average, loss_aft_window.average,
				# 																																		 loss_stop_window.average, loss_reg_window.average,
				# 																																		 loss_emt_window.average, loss_spk_window.average,
				# 																																		 loss_orthog_window.average)
				# if args.unpaired:
				# 	vars += [model.style_emb_loss_up_emt, model.style_emb_loss_up_spk,model.style_emb_loss_mel_out_up_emt, model.style_emb_loss_mel_out_up_spk]
				# 	out += [loss_up_emt, loss_up_spk, loss_mo_up_emt, loss_mo_up_spk]
				# 	message += ' up_emt={:.5f}, up_spk={:.5f}, mo_up_emt={:.5f}, mo_up_spk={:.5f}]'.format(loss_up_emt_window.average,
				# 																																												loss_up_spk_window.average,
				# 																																												loss_mo_up_emt_window.average,
				# 																																												loss_mo_up_spk_window.average)
				# if False:
				# 	vars += [model.tower_style_emb_logit_emt[0], model.tower_emt_labels[0],model.tower_style_emb_logit_up_emt[0],
				# 					model.tower_emt_up_labels[0],model.tower_spk_labels[0]]
				# 	out += [emt_logit, emt_labels, emt_up_logit, emt_up_labels, spk_labels]
				#
				# out = sess.run([vars])

				if args.nat_gan and (args.restart_nat_gan_d or not(args.restore)) and step ==0:
					log("Will start with Training Nat GAN Discriminator", end='\r')
					disc_epochs = 300 if args.unpaired else 200
					disc_epochs = 0 if args.TEST else disc_epochs
					for i in range(disc_epochs+1):
						d_loss_t, d_loss_p, d_loss_up,\
						d_loss_t_emt, d_loss_p_emt, d_loss_up_emt, \
						d_loss_t_spk, d_loss_p_spk, d_loss_up_spk, \
						opt_n = sess.run([model.d_loss_targ, model.d_loss_p, model.d_loss_up,
																														 model.d_loss_targ_emt, model.d_loss_p_emt, model.d_loss_up_emt,
																														 model.d_loss_targ_spk, model.d_loss_p_spk, model.d_loss_up_spk,
																														 model.optimize_n])
						message = 'step: {}, d_loss_t={:.5f}, d_loss_p ={:.5f}, d_loss_up ={:.5f},' \
											' d_loss_t_emt={:.5f}, d_loss_p_emt ={:.5f}, d_loss_up_emt ={:.5f},' \
											' d_loss_t_spk={:.5f}, d_loss_p_spk ={:.5f}, d_loss_up_spk ={:.5f}'.format(i, d_loss_t, d_loss_p, d_loss_up,
																																															d_loss_t_emt, d_loss_p_emt, d_loss_up_emt,
																																															d_loss_t_spk, d_loss_p_spk, d_loss_up_spk)
						log(message, end='\r')
					os.makedirs(r'nat_gan',exist_ok=True)
					os.makedirs(r'nat_gan/pretrained_model', exist_ok=True)
					checkpoint_path_nat_gan = os.path.join(save_dir_nat_gan, 'nat_gan_model.ckpt')
					saver_nat_gan.save(sess, checkpoint_path_nat_gan, global_step=i)

				if args.nat_gan:
					d_loss_t, d_loss_p, d_loss_up, opt_n = sess.run([model.d_loss_targ, model.d_loss_p, model.d_loss_up, model.optimize_n])

				if args.unpaired:
					step, tfr, loss, opt, bef, aft, stop, reg, loss_emt, loss_spk, loss_orthog, \
					loss_up_emt, loss_up_spk, loss_mo_up_emt, loss_mo_up_spk, g_loss_p, g_loss_up, mels, opt_r\
					= sess.run([global_step, model.ratio, model.loss, model.optimize,model.before_loss, model.after_loss,model.stop_token_loss,
									model.regularization_loss, model.style_emb_loss_emt, model.style_emb_loss_spk, model.style_emb_orthog_loss,
									model.style_emb_loss_up_emt, model.style_emb_loss_up_spk,model.style_emb_loss_mel_out_up_emt,
											model.style_emb_loss_mel_out_up_spk,model.g_loss_p, model.g_loss_up, model.tower_mel_outputs[0], model.optimize_r])

				else:
					step, tfr, loss, opt, bef, aft, stop, reg, loss_emt, loss_spk, loss_orthog, \
					loss_up_emt, loss_up_spk, loss_mo_up_emt, loss_mo_up_spk, g_loss_p, g_loss_up, mels,dec_out,opt_r = sess.run([global_step, model.helper._ratio, model.loss,
									model.optimize, model.before_loss, model.after_loss, model.stop_token_loss,
									model.regularization_loss, model.style_emb_loss_emt, model.style_emb_loss_spk, model.style_emb_orthog_loss,
									model.style_emb_loss_up_emt, model.style_emb_loss_up_spk,model.style_emb_loss_mel_out_up_emt,
									model.style_emb_loss_mel_out_up_spk, model.g_loss_p, model.g_loss_up, model.tower_mel_outputs[0],model.tower_decoder_output[0],model.optimize_r])


					# step, loss, opt, bef, aft, stop, reg, loss_emt, loss_spk, loss_orthog, \
					# loss_up_emt, loss_up_spk, loss_mo_up_emt, loss_mo_up_spk, g_loss_p, g_loss_up, mels,ref_emt,ref_spk,ref_up_emt,ref_up_spk,emb,enc_out,enc_out_up,\
					# stop_pred, targ, inp, inp_len,targ_len,stop_targ,mels_up,dec_out,dec_out_up,opt_r\
					# = sess.run([global_step, model.loss, model.optimize,model.before_loss, model.after_loss,model.stop_token_loss,
					# 				model.regularization_loss, model.style_emb_loss_emt, model.style_emb_loss_spk, model.style_emb_orthog_loss,
					# 				model.style_emb_loss_up_emt, model.style_emb_loss_up_spk,model.style_emb_loss_mel_out_up_emt,
					# 						model.style_emb_loss_mel_out_up_spk,model.g_loss_p, model.g_loss_up, model.tower_mel_outputs[0],
					# 						model.tower_refnet_out_emt[0],model.tower_refnet_out_spk[0],model.tower_refnet_out_up_emt[0],model.tower_refnet_out_up_spk[0],
					# 						model.tower_embedded_inputs[0], model.tower_encoder_outputs[0],model.tower_encoder_outputs_up[0],model.tower_stop_token_prediction[0],
					# 						model.tower_mel_targets[0],model.tower_inputs[0],model.tower_input_lengths[0],model.tower_targets_lengths[0],
					# 						model.tower_stop_token_targets[0],model.tower_mel_outputs_up[0],model.tower_decoder_output[0],model.tower_decoder_output_up[0],model.optimize_r])
					#
					# if args.save_output_vars:
					# 	import pandas as pd
					# 	pd.DataFrame(emb[:, 0, :]).to_csv(r'C:\Users\t-mawhit\Documents\code\Tacotron-2\eval\mels_save\emb.csv')
					# 	pd.DataFrame(enc_out[:, 0, :]).to_csv(r'C:\Users\t-mawhit\Documents\code\Tacotron-2\eval\mels_save\enc_out.csv')
					# 	pd.DataFrame(enc_out_up[:, 0, :]).to_csv(r'C:\Users\t-mawhit\Documents\code\Tacotron-2\eval\mels_save\enc_out_up.csv')
					# 	pd.DataFrame(stop_pred[:, :]).to_csv(r'C:\Users\t-mawhit\Documents\code\Tacotron-2\eval\mels_save\stop.csv')
					# 	pd.DataFrame(targ[:, 0, :]).to_csv(r'C:\Users\t-mawhit\Documents\code\Tacotron-2\eval\mels_save\targ.csv')
					# 	pd.DataFrame(inp[:, :]).to_csv(r'C:\Users\t-mawhit\Documents\code\Tacotron-2\eval\mels_save\inp.csv')
					# 	pd.DataFrame(inp_len[:]).to_csv(r'C:\Users\t-mawhit\Documents\code\Tacotron-2\eval\mels_save\inp_len.csv')
					# 	pd.DataFrame(targ_len[:]).to_csv(r'C:\Users\t-mawhit\Documents\code\Tacotron-2\eval\mels_save\targ_len.csv')
					# 	pd.DataFrame(stop_targ[:, :]).to_csv(r'C:\Users\t-mawhit\Documents\code\Tacotron-2\eval\mels_save\stop_targ.csv')
					# 	pd.DataFrame(mels_up[:, 0, 0:5]).to_csv(r'C:\Users\t-mawhit\Documents\code\Tacotron-2\eval\mels_save\mels_up.csv')
					# 	pd.DataFrame(dec_out_up[:, 0, 0:5]).to_csv(r'C:\Users\t-mawhit\Documents\code\Tacotron-2\eval\mels_save\dec_out_up.csv')


					if args.save_output_vars:
						import pandas as pd
						pd.DataFrame(mels[:, 0, 0:5]).to_csv(r'C:\Users\t-mawhit\Documents\code\Tacotron-2\eval\mels_save\mels.csv')
						pd.DataFrame(dec_out[:, 0, 0:5]).to_csv(r'C:\Users\t-mawhit\Documents\code\Tacotron-2\eval\mels_save\dec_out.csv')

				# import pandas as pd
				# print(emt_logit.shape, emt_labels.shape)
				# if len(emt_logit.shape)>2:
				# 	emt_logit = emt_logit.squeeze(1)
				# 	emt_up_logit = emt_up_logit.squeeze(1)
				# emt_labels = emt_labels.reshape(-1,1)
				# emt_up_labels = emt_up_labels.reshape(-1, 1)
				# spk_labels = spk_labels.reshape(-1, 1)
				# df = np.concatenate((emt_logit,emt_labels,spk_labels,emt_up_logit,emt_up_labels),axis=1)
				# print(emt_labels)
				# print(emt_logit)
				# print(emt_up_labels)
				# print(emt_up_logit)
				#
				# pd.DataFrame(df).to_csv(r'C:\Users\t-mawhit\Documents\code\Tacotron-2\eval\mels_save\emt_logit_.001_up_10k.csv')
				# raise

				time_window.append(time.time() - start_time)
				loss_window.append(loss)
				loss_bef_window.append(bef)
				loss_aft_window.append(aft)
				loss_stop_window.append(stop)
				loss_reg_window.append(reg)
				loss_emt_window.append(loss_emt)
				loss_spk_window.append(loss_spk)
				loss_orthog_window.append(loss_orthog)
				loss_up_emt_window.append(loss_up_emt)
				loss_up_spk_window.append(loss_up_spk)
				loss_mo_up_emt_window.append(loss_mo_up_emt)
				loss_mo_up_spk_window.append(loss_mo_up_spk)

				if args.nat_gan:
					d_loss_t_window.append(d_loss_t)
					d_loss_p_window.append(d_loss_p)
					d_loss_up_window.append(d_loss_up)
					g_loss_p_window.append(g_loss_p)
					g_loss_up_window.append(g_loss_up)

				message = 'Step {:7d} {:.3f} sec/step, tfr={:.3f}, loss={:.5f}, avg_loss={:.5f}, bef={:.5f}, aft={:.5f}, stop={:.5f}, reg={:.5f}'.format(step,time_window.average,tfr, loss,loss_window.average,
																															loss_bef_window.average, loss_aft_window.average,
																															loss_stop_window.average, loss_reg_window.average)
				if args.emt_attn:
					message += ' emt={:.5f}, spk={:.5f}, spk_l2={:.5f}'.format(loss_emt_window.average, loss_spk_window.average, loss_orthog_window.average)
				else:
					message += ' emt={:.5f}, spk={:.5f}, orthog={:.5f},'.format(loss_emt_window.average, loss_spk_window.average,
																																			loss_orthog_window.average)
				if args.unpaired:
					message += ' up_emt={:.5f}, up_spk={:.5f}, mo_up_emt={:.5f}, mo_up_spk={:.5f}'.format(loss_up_emt_window.average,
																																																 loss_up_spk_window.average,
																																																 loss_mo_up_emt_window.average,
																																																 loss_mo_up_spk_window.average)
				if args.nat_gan:
					message += ' d_loss_t={:.5f}, d_loss_p ={:.5f}, d_loss_up ={:.5f}, g_loss_p ={:.5f}, g_loss_up ={:.5f}'.format(
						d_loss_t_window.average, d_loss_p_window.average, d_loss_up_window.average, g_loss_p_window.average, g_loss_up_window.average)

				log(message, end='\r', slack=(step % args.checkpoint_interval == 0))

				if np.isnan(loss) or loss > 100.:
					log('Loss exploded to {:.5f} at step {}'.format(loss, step))
					raise Exception('Loss exploded')

				if step % args.summary_interval == 0:
					log('\nWriting summary at step {}'.format(step))
					summary_writer.add_summary(sess.run(stats), step)

				# if step % args.eval_interval == 0:
				# 	#Run eval and save eval stats
				# 	log('\nRunning evaluation and saving model at step {}'.format(step))
				# 	saver.save(sess, checkpoint_path, global_step=global_step)
				#
				# 	eval_losses = []
				# 	before_losses = []
				# 	after_losses = []
				# 	stop_token_losses = []
				# 	linear_losses = []
				# 	linear_loss = None
				#
				# 	if hparams.predict_linear:
				# 		for i in tqdm(range(feeder.test_steps)):
				# 			eloss, before_loss, after_loss, stop_token_loss, linear_loss, mel_p, mel_t, t_len, align, lin_p, lin_t = sess.run([
				# 				eval_model.tower_loss[0], eval_model.tower_before_loss[0], eval_model.tower_after_loss[0],
				# 				eval_model.tower_stop_token_loss[0], eval_model.tower_linear_loss[0], eval_model.tower_mel_outputs[0][0],
				# 				eval_model.tower_mel_targets[0][0], eval_model.tower_targets_lengths[0][0],
				# 				eval_model.tower_alignments[0][0], eval_model.tower_linear_outputs[0][0],
				# 				eval_model.tower_linear_targets[0][0],
				# 				])
				# 			eval_losses.append(eloss)
				# 			before_losses.append(before_loss)
				# 			after_losses.append(after_loss)
				# 			stop_token_losses.append(stop_token_loss)
				# 			linear_losses.append(linear_loss)
				# 		linear_loss = sum(linear_losses) / len(linear_losses)
				#
				# 		if hparams.GL_on_GPU:
				# 			wav = sess.run(GLGPU_lin_outputs, feed_dict={GLGPU_lin_inputs: lin_p})
				# 			wav = audio.inv_preemphasis(wav, hparams.preemphasis, hparams.preemphasize)
				# 		else:
				# 			wav = audio.inv_linear_spectrogram(lin_p.T, hparams)
				# 		audio.save_wav(wav, os.path.join(eval_wav_dir, 'step-{}-eval-wave-from-linear.wav'.format(step)), sr=hparams.sample_rate)
				#
				# 	else:
				# 		for i in tqdm(range(feeder.test_steps)):
				# 			eloss, before_loss, after_loss, stop_token_loss, input_seq, mel_p, mel_t, t_len, align = sess.run([
				# 				eval_model.tower_loss[0], eval_model.tower_before_loss[0], eval_model.tower_after_loss[0],
				# 				eval_model.tower_stop_token_loss[0],eval_model.tower_inputs[0][0], eval_model.tower_mel_outputs[0][0],
				# 				eval_model.tower_mel_targets[0][0],
				# 				eval_model.tower_targets_lengths[0][0], eval_model.tower_alignments[0][0]
				# 				])
				# 			eval_losses.append(eloss)
				# 			before_losses.append(before_loss)
				# 			after_losses.append(after_loss)
				# 			stop_token_losses.append(stop_token_loss)
				#
				# 	eval_loss = sum(eval_losses) / len(eval_losses)
				# 	before_loss = sum(before_losses) / len(before_losses)
				# 	after_loss = sum(after_losses) / len(after_losses)
				# 	stop_token_loss = sum(stop_token_losses) / len(stop_token_losses)
				#
				# 	# log('Saving eval log to {}..'.format(eval_dir))
				# 	#Save some log to monitor model improvement on same unseen sequence
				# 	if hparams.GL_on_GPU:
				# 		wav = sess.run(GLGPU_mel_outputs, feed_dict={GLGPU_mel_inputs: mel_p})
				# 		wav = audio.inv_preemphasis(wav, hparams.preemphasis, hparams.preemphasize)
				# 	else:
				# 		wav = audio.inv_mel_spectrogram(mel_p.T, hparams)
				# 	audio.save_wav(wav, os.path.join(eval_wav_dir, 'step-{}-eval-wave-from-mel.wav'.format(step)), sr=hparams.sample_rate)
				#
				# 	input_seq = sequence_to_text(input_seq)
				# 	plot.plot_alignment(align, os.path.join(eval_plot_dir, 'step-{}-eval-align.png'.format(step)),
				# 		title='{}, {}, step={}, loss={:.5f}\n{}'.format(args.model, time_string(), step, eval_loss, input_seq),
				# 		max_len=t_len // hparams.outputs_per_step)
				# 	plot.plot_spectrogram(mel_p, os.path.join(eval_plot_dir, 'step-{}-eval-mel-spectrogram.png'.format(step)),
				# 		title='{}, {}, step={}, loss={:.5f}\n{}'.format(args.model, time_string(), step, eval_loss,input_seq), target_spectrogram=mel_t,
				# 		max_len=t_len)
				#
				# 	if hparams.predict_linear:
				# 		plot.plot_spectrogram(lin_p, os.path.join(eval_plot_dir, 'step-{}-eval-linear-spectrogram.png'.format(step)),
				# 			title='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step, eval_loss), target_spectrogram=lin_t,
				# 			max_len=t_len, auto_aspect=True)
				#
				# 	log('Step {:7d} [eval loss: {:.3f}, before loss: {:.3f}, after loss: {:.3f}, stop loss: {:.3f}]'.format(step, eval_loss, before_loss, after_loss, stop_token_loss))
				# 	# log('Writing eval summary!')
				# 	add_eval_stats(summary_writer, step, linear_loss, before_loss, after_loss, stop_token_loss, eval_loss)


				if step % args.checkpoint_interval == 0 or step == args.tacotron_train_steps or step == 300:
					#Save model and current global step
					saver.save(sess, checkpoint_path, global_step=global_step)

					log('\nSaved model at step {}'.format(step))

				if step % args.eval_interval == 0:

					if hparams.predict_linear:
						raise ValueError('predict linear not implemented')
						# input_seq, mel_prediction, linear_prediction, alignment, target, target_length, linear_target = sess.run([
						# 	model.tower_inputs[0][0],
						# 	model.tower_mel_outputs[0][0],
						# 	model.tower_linear_outputs[0][0],
						# 	model.tower_alignments[0][0],
						# 	model.tower_mel_targets[0][0],
						# 	model.tower_targets_lengths[0][0],
						# 	model.tower_linear_targets[0][0],
						# 	])
						#
						# #save predicted linear spectrogram to disk (debug)
						# linear_filename = 'linear-prediction-step-{}.npy'.format(step)
						# np.save(os.path.join(linear_dir, linear_filename), linear_prediction.T, allow_pickle=False)
						#
						# #save griffin lim inverted wav for debug (linear -> wav)
						# if hparams.GL_on_GPU:
						# 	wav = sess.run(GLGPU_lin_outputs, feed_dict={GLGPU_lin_inputs: linear_prediction})
						# 	wav = audio.inv_preemphasis(wav, hparams.preemphasis, hparams.preemphasize)
						# else:
						# 	wav = audio.inv_linear_spectrogram(linear_prediction.T, hparams)
						# audio.save_wav(wav, os.path.join(wav_dir, 'step-{}-wave-from-linear.wav'.format(step)), sr=hparams.sample_rate)
						#
						# #Save real and predicted linear-spectrogram plot to disk (control purposes)
						# plot.plot_spectrogram(linear_prediction, os.path.join(plot_dir, 'step-{}-linear-spectrogram.png'.format(step)),
						# 	title='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step, loss), target_spectrogram=linear_target,
						# 	max_len=target_length, auto_aspect=True)

					else:
						input_seqs, mels, alignments,\
						stop_tokens = sess.run([eval_model.tower_inputs,
												eval_model.tower_mel_outputs,
												eval_model.tower_alignments,
												eval_model.tower_stop_token_prediction],
											    feed_dict=eval_feed_dict)

						# num_evals = len(input_seqs) if False else 1
						# for i in range(num_evals):
						# 	input_seq = input_seqs[i]
						# 	mel_prediction = mel_predictions[i]
						# 	alignment = alignments[i]
						# 	target = targets[i]
						# 	target_length = target_lengths[i]
						# 	emt = emts[i]
						# 	spk = spks[i]
						# 	if args.emt_attn and args.attn=='simple':
						# 		alignment_emt = alignments_emt[0][i]

						# Linearize outputs (n_gpus -> 1D)
						inp = [inp for gpu_inp in input_seqs for inp in gpu_inp]
						mels = [mel for gpu_mels in mels for mel in gpu_mels]
						# targets = [target for gpu_targets in targets for target in gpu_targets]
						alignments = [align for gpu_aligns in alignments for align in gpu_aligns]
						stop_tokens = [token for gpu_token in stop_tokens for token in gpu_token]

						target_lengths = get_output_lengths(stop_tokens)

						# Take off the batch wise padding
						mels = [mel[:target_length, :] for mel, target_length in zip(mels, target_lengths)]

						T2_output_range = (-hparams.max_abs_value, hparams.max_abs_value) if hparams.symmetric_mels else (0, hparams.max_abs_value)
						mels = [np.clip(m, T2_output_range[0], T2_output_range[1]) for m in mels]

						folder_bucket = 'step_{}'.format(step//500)
						folder_wavs_save = os.path.join(wav_dir,folder_bucket)
						folder_plot_save = os.path.join(plot_dir,folder_bucket)
						os.makedirs(folder_wavs_save, exist_ok=True)
						os.makedirs(folder_plot_save, exist_ok=True)

						for i, (mel,align,basename,basename_ref) in enumerate(zip(mels, alignments,basenames, basenames_refs)):

							#save griffin lim inverted wav for debug (mel -> wav)
							if hparams.GL_on_GPU:
								wav = sess.run(GLGPU_mel_outputs, feed_dict={GLGPU_mel_inputs: mel})
								wav = audio.inv_preemphasis(wav, hparams.preemphasis, hparams.preemphasize)
							else:
								wav = audio.inv_mel_spectrogram(mel.T, hparams)
							audio.save_wav(wav, os.path.join(folder_wavs_save, 'step_{}_wav_{}_{}_{}.wav'.format(step, i, basename, basename_ref)),
										   																		 sr=hparams.sample_rate)

							input_seq = sequence_to_text(inp[i])
							#save alignment plot to disk (control purposes)
							try:
								plot.plot_alignment(align, os.path.join(folder_plot_save, 'step_{}_wav_{}_{}_{}_align.png'.format(step, i, basename, basename_ref)),
									title='{}, {}, step={}\n{}'.format(args.model, time_string(), step, input_seq),
									max_len=target_lengths[i] // hparams.outputs_per_step)
							except:
								print("failed to plot alignment")
							try:
								#save real and predicted mel-spectrogram plot to disk (control purposes)
								plot.plot_spectrogram(mel, os.path.join(folder_plot_save, 'step-{}-{}-mel-spectrogram.png'.format(step, i)),
													  title='{}, {}, step={}\n{}'.format(args.model, time_string(), step, input_seq))
													  # target_spectrogram=targets[i],
													  # max_len=target_lengths[i])
							except:
								print("failed to plot spectrogram")

						log('Saved synthesized samples for step {}'.format(step), end='\r')
						# log('Input at step {}: {}'.format(step, input_seq), end='\r')

				# if step % args.embedding_interval == 0 or step == args.tacotron_train_steps or step == 1:
				# 	#Get current checkpoint state
				# 	checkpoint_state = tf.train.get_checkpoint_state(save_dir)
				#
				# 	#Update Projector
				# 	log('\nSaving Model Character Embeddings visualization..')
				# 	add_embedding_stats(summary_writer, [model.embedding_table.name], [char_embedding_meta], checkpoint_state.model_checkpoint_path)
				# 	log('Tacotron Character embeddings have been updated on tensorboard!')

			log('Tacotron training complete after {} global steps!'.format(args.tacotron_train_steps), slack=True)
			return save_dir

		except Exception as e:
			log('Exiting due to exception: {}'.format(e), slack=True)
			traceback.print_exc()
			coord.request_stop(e)

def tacotron_train(args, log_dir, hparams):
	return train(log_dir, args, hparams)