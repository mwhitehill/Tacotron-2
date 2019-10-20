import argparse
import os
import re
import time
from time import sleep
import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import datetime

if __name__ == '__main__':
	import sys
	sys.path.append(os.getcwd())
	import argparse

from hparams import hparams, hparams_debug_string
from infolog import log
from tacotron.synthesizer import Synthesizer
from tqdm import tqdm
from tacotron.feeder import get_metadata_df

def time_string():
	return datetime.now().strftime('%Y.%m.%d_%H-%M-%S')

def generate_fast(model, text):
	model.synthesize([text], None, None, None, None)


def run_live(args, checkpoint_path, hparams):
	#Log to Terminal without keeping any records in files
	log(hparams_debug_string())
	synth = Synthesizer()
	synth.load(checkpoint_path, hparams)

	#Generate fast greeting message
	greetings = 'Hello, Welcome to the Live testing tool. Please type a message and I will try to read it!'
	log(greetings)
	generate_fast(synth, greetings)

	#Interaction loop
	while True:
		try:
			text = input()
			generate_fast(synth, text)

		except KeyboardInterrupt:
			leave = 'Thank you for testing our features. see you soon.'
			log(leave)
			generate_fast(synth, leave)
			sleep(2)
			break

def run_eval(args, checkpoint_path, output_dir, hparams, sentences):
	eval_dir = os.path.join(output_dir, 'eval')
	log_dir = os.path.join(output_dir, 'logs-eval')

	if args.model == 'Tacotron-2':
		assert os.path.normpath(eval_dir) == os.path.normpath(args.mels_dir)

	#Create output path if it doesn't exist
	os.makedirs(eval_dir, exist_ok=True)
	os.makedirs(log_dir, exist_ok=True)
	os.makedirs(os.path.join(log_dir, 'wavs'), exist_ok=True)
	os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)

	log(hparams_debug_string())
	synth = Synthesizer()
	synth.load(checkpoint_path, hparams)

	#Set inputs batch wise
	sentences = [sentences[i: i+hparams.tacotron_synthesis_batch_size] for i in range(0, len(sentences), hparams.tacotron_synthesis_batch_size)]

	log('Starting Synthesis')
	with open(os.path.join(eval_dir, 'map.txt'), 'w') as file:
		for i, texts in enumerate(tqdm(sentences)):
			start = time.time()
			basenames = ['batch_{}_sentence_{}'.format(i, j) for j in range(len(texts))]
			mel_filenames, speaker_ids = synth.synthesize(texts, basenames, eval_dir, log_dir, None)

			for elems in zip(texts, mel_filenames, speaker_ids):
				file.write('|'.join([str(x) for x in elems]) + '\n')
	log('synthesized mel spectrograms at {}'.format(eval_dir))
	return eval_dir

def get_filenames_from_metadata(synth_metadata_filename, input_dir, flip_spk_emt=False):

	with open(synth_metadata_filename, encoding='utf-8') as f:
		metadata = [line.strip().split('|') for line in f if not(line.startswith('#'))]
		frame_shift_ms = hparams.hop_size / hparams.sample_rate
		hours = sum([int(x[6]) for x in metadata]) * frame_shift_ms / (3600)
		log('Synthesis - Loaded metadata for {} examples ({:.2f} hours)'.format(len(metadata), hours))

	# log('Starting Synthesis')
	texts = [m[7] for m in metadata]
	mel_filenames = [os.path.join(input_dir, m[0], 'mels', m[2]) for m in metadata]
	basenames = [os.path.basename(m).replace('.npy', '').replace('mel-', '') for m in mel_filenames]
	basenames_refs = [m[11]+'_'+m[13] for m in metadata]

	mel_ref_filenames_emt = []
	mel_ref_filenames_spk = []
	emt_labels = []
	spk_labels = []
	for m in metadata:
		dataset = m[0]
		if m[12] == 'same':
			mel_ref_filenames_emt.append(os.path.join(input_dir, dataset, 'mels', m[2]))
		else:
			if 'accent' in synth_metadata_filename:
				dataset_emt = 'vctk'
			else:
				dataset_emt = 'emth' if m[12][0] == 'h' else 'emt4'
				if m[12][0] == 'h':
					m[12] = m[12][1:]
			mel_ref_filenames_emt.append(os.path.join(input_dir, dataset_emt , 'mels', m[12]))

		if m[14] == 'same':
			mel_ref_filenames_spk.append(os.path.join(input_dir, dataset,'mels', m[2]))
		else:
			mel_ref_filenames_spk.append(os.path.join(input_dir, 'jessa', 'mels', m[14]))
		emt_labels.append(m[8])
		spk_labels.append(m[9])
	if flip_spk_emt:
		mel_ref_filenames_emt_tmp = mel_ref_filenames_emt
		mel_ref_filenames_emt = mel_ref_filenames_spk
		mel_ref_filenames_spk = mel_ref_filenames_emt_tmp

	return(texts, basenames, basenames_refs, mel_filenames, mel_ref_filenames_emt, mel_ref_filenames_spk,
		   emt_labels, spk_labels)


def run_synthesis_sytle_transfer(args, synth_metadata_filename, checkpoint_path, output_dir, hparams):

	synth_dir = os.path.join(output_dir, 'natural')

	#Create output path if it doesn't exist
	os.makedirs(synth_dir, exist_ok=True)

	log(hparams_debug_string())
	synth = Synthesizer()
	synth.load(args, checkpoint_path, hparams)

	texts, basenames, basenames_refs, mel_filenames, \
	mel_ref_filenames_emt, mel_ref_filenames_spk,\
	emt_labels, spk_labels = get_filenames_from_metadata(synth_metadata_filename, args.input_dir, args.flip_spk_emt)

	synth.synthesize(texts, basenames, synth_dir,synth_dir, mel_filenames,
					 mel_ref_filenames_emt=mel_ref_filenames_emt,
					 mel_ref_filenames_spk=mel_ref_filenames_spk,
					 emt_labels_synth=emt_labels,spk_labels_synth=spk_labels)

def synthesize_random(args, checkpoint_path, output_dir,
											hparams, model_suffix):

	n_emt = 4 if not(args.paired) else 1
	n_txts_per_emotion = 5 if not(args.paired) else 10

	synth_dir = os.path.join(output_dir, 'random', model_suffix, time_string())
	os.makedirs(synth_dir, exist_ok=True)

	synth = Synthesizer()
	synth.load(args, checkpoint_path, hparams)

	meta_save_path = os.path.join(synth_dir, 'meta.csv')

	df = pd.read_csv(r'C:\Users\t-mawhit\Documents\code\Tacotron-2\data\zo_jessa_train_test.csv')
	df_train = df[df.train_test == 'train']
	df_test = df[df.train_test == 'test']

	#synthesize 20 random samples from zo and jessa, 5 in each emotion
	#change emotion

	df_test_zo = df_test[df_test.dataset == 'emt4']
	df_test_jessa = df_test[df_test.dataset == 'jessa']

	df_test_use = df_test_jessa if not(args.zo) else df_test_zo[df_test_zo.emt_label==0]

	np.random.seed(2)
	chosen_texts_idxs = np.random.choice(df_test_use.index, n_txts_per_emotion * n_emt, replace=False)
	df_test_use_texts_rows =df_test_use.loc[chosen_texts_idxs]
	meta = df_test_use_texts_rows.copy()
	meta['basename'] = ''
	idx = 0

	texts = []
	mel_filenames = []
	mel_ref_filenames_emt = []
	mel_ref_filenames_spk = []
	basenames = []
	basenames_refs = []
	emt_labels=[]
	spk_labels=[]

	for i in range(n_emt):
		df_test_zo_emt = df_test_zo[df_test_zo.emt_label ==i]
		for j in range(n_txts_per_emotion):
			row = df_test_use_texts_rows.iloc[idx]
			texts.append(row.text)
			mel_filenames.append(os.path.join(args.input_dir, row.dataset, 'mels', row.mel_filename))

			if args.paired:
				mel_ref_filenames_spk.append(os.path.join(args.input_dir, row.dataset, 'mels', row.mel_filename))
				mel_ref_filenames_emt.append(os.path.join(args.input_dir, row.dataset, 'mels', row.mel_filename))
			else:
				row_spk = df_test_use.loc[np.random.choice(df_test_use.index)]
				mel_ref_filenames_spk.append(os.path.join(args.input_dir, row_spk.dataset, 'mels', row_spk.mel_filename))

				row_emt = df_test_zo_emt.loc[np.random.choice(df_test_zo_emt.index)]
				mel_ref_filenames_emt.append(os.path.join(args.input_dir, row_emt.dataset, 'mels', row_emt.mel_filename))

			basename = '{}'.format(row.basename.split('.')[0])
			basename_ref = 'e{}'.format(i)

			basenames.append(basename)
			basenames_refs.append(basename_ref)

			emt_label = row_emt.emt_label if not(args.paired) else row.emt_label
			spk_label = row_spk.spk_label if not(args.paired) else row.spk_label
			emt_labels.append(int(emt_label))
			spk_labels.append(int(spk_label))
			meta.iloc[idx,8] = emt_label
			meta.iloc[idx,9] = spk_label
			meta.iloc[idx,10] = 'mel-{}_{}.npy'.format(basename,basename_ref)

			idx+=1

	meta.to_csv(meta_save_path,index=False)

	print('Starting Synthesis on {} samples'.format(len(mel_filenames)))
	synth.synthesize(texts, basenames, synth_dir, synth_dir, mel_filenames, basenames_refs=basenames_refs,
									 mel_ref_filenames_emt=mel_ref_filenames_emt, mel_ref_filenames_spk=mel_ref_filenames_spk,
									 emt_labels_synth=emt_labels, spk_labels_synth=spk_labels)

def run_synthesis_multiple(args, checkpoint_path, output_dir, hparams, model_suffix):

	n_spk_per_accent = 2
	n_text_per_spk = 5

	synth_dir = os.path.join(output_dir, 'wavs', model_suffix, time_string())
	os.makedirs(synth_dir, exist_ok=True)

	synth = Synthesizer()
	synth.load(args, checkpoint_path, hparams)

	with open(args.train_filename, encoding='utf-8') as f:
		metadata = [line.strip().split('|') for line in f]
		if args.remove_long_samps:
			len_before = len(metadata)
			metadata = [f for f in metadata if not (f[10].endswith('_023.wav'))]
			metadata = [f for f in metadata if not (f[10].endswith('_021.wav'))]
			metadata = [f for f in metadata if int(f[6]) < 500]
			print("Removed Long Samples - before: {}, after: {}".format(len_before,len(metadata)))

		#only synthesize long samples
		metadata = [f for f in metadata if int(f[6]) >200]

		frame_shift_ms = hparams.hop_size / hparams.sample_rate
		hours = sum([int(x[6]) for x in metadata]) * frame_shift_ms / (3600)
		print('Loaded metadata for {} examples ({:.2f} hours)'.format(len(metadata), hours))

	df = pd.DataFrame(metadata,	columns = ['dataset','audio_filename', 'mel_filename', 'linear_filename', 'spk_emb_filename', 'time_steps', 'mel_frames', 'text',
						 'emt_label', 'spk_label', 'basename', 'sex'])
	chosen_accents = ['0','3']
	assert (len(chosen_accents) <= 2)
	acc_names = ['American','Australian','Canadian','English','Indian','Irish','NewZealand','NorthernIrish',
							 'Scottish','SouthAfrican','Welsh']
	df_acc = df[df['emt_label'].isin(chosen_accents)]
	# spk_idxs = sorted(frozenset(df_acc['spk_label'].unique()))
	texts = []
	mel_filenames = []
	mel_ref_filenames_emt =[]
	mel_ref_filenames_spk = []
	basenames = []
	basenames_refs = []

	for i,acc in enumerate(chosen_accents):
		df_acc_spks = df_acc[df_acc['emt_label']==acc]['spk_label'].unique()
		chosen_spks = np.random.choice(df_acc_spks,n_spk_per_accent,replace=False)

		for spk in chosen_spks:
			df_spk = df_acc[df_acc['spk_label']==spk]
			idxs = np.random.choice(df_spk.index, n_text_per_spk, replace=False)
			for idx in idxs:
				# for j in range(5):
				for acc_ref in chosen_accents:
					texts.append(df_acc.loc[idx].text)
					mel_filename = os.path.join(args.input_dir, df_acc.loc[idx].dataset, 'mels', df_acc.loc[idx].mel_filename)
					mel_filenames.append(mel_filename)
					mel_ref_filenames_spk.append(mel_filename)
					basenames.append('{}_{}_{}'.format(df_acc.loc[idx].basename.split('.')[0],
																						 acc_names[int(acc)][:2],df_acc.loc[idx].sex))

					df_other_acc = df_acc[df_acc['emt_label']==acc_ref]
					row = df_other_acc.loc[np.random.choice(df_other_acc.index, 1)]
					mel_ref_filenames_emt.append(os.path.join(args.input_dir, row.dataset.iloc[0], 'mels',
																										row.mel_filename.iloc[0]))
					basenames_refs.append('{}'.format(acc_names[int(row.emt_label)][:2]))#,j))

	if args.flip_spk_emt:
		mel_ref_filenames_emt_tmp = mel_ref_filenames_emt
		mel_ref_filenames_emt = mel_ref_filenames_spk
		mel_ref_filenames_spk = mel_ref_filenames_emt_tmp

	print('Starting Synthesis on {} samples'.format(len(mel_filenames)//len(chosen_accents)))
	synth.synthesize(texts, basenames, synth_dir, synth_dir, mel_filenames, basenames_refs=basenames_refs,
									 mel_ref_filenames_emt=mel_ref_filenames_emt,mel_ref_filenames_spk=mel_ref_filenames_spk)

def get_style_embeddings(args, checkpoint_path, output_dir, hparams):

	emb_dir = os.path.join(output_dir, 'embeddings')
	os.makedirs(emb_dir, exist_ok=True)
	meta_path = os.path.join(emb_dir,'meta.tsv')
	emb_emt_path = os.path.join(emb_dir,'emb_emt.tsv')
	emb_spk_path = os.path.join(emb_dir,'emb_spk.tsv')

	with open(args.train_filename , encoding='utf-8') as f:
		metadata = [line.strip().split('|') for line in f if not(line.startswith('#'))]

	df_meta = get_metadata_df(args.train_filename, args)

	spk_ids = df_meta.spk_label.unique()
	spk_ids_chosen = np.sort(np.random.choice(spk_ids,args.n_spk))

	#make sure first user is in embeddings (zo - the one with emotions)
	# if not(0 in spk_ids_chosen):
	# 	spk_ids_chosen = np.sort(np.append(spk_ids_chosen,0))


	# if args.unpaired:
	# 	chosen_idx = []
	# 	for id in spk_ids_chosen:
	# 		spk_rows = df_meta[df_meta.loc[:, 'spk_label'] == id]
	# 		chosen_idxs  = np.random.choice(spk_rows.index.values, args.n_per_spk)
	# 		for idx in chosen_idxs:
	# 			row = df_meta
	# 			for i in range(4):
	# 				if i ==0:
	#
	#
	# 	df_meta_chosen = df_meta.iloc[np.array(sorted(chosen_idx))]
	#
	# 	mel_filenames = [os.path.join(args.input_dir, row.dataset, 'mels', row.mel_filename) for idx, row in
	# 									 df_meta_chosen.iterrows()]
	#
	#
	# 	texts = list(df_meta_chosen.text)


	chosen_idx = []
	for id in spk_ids_chosen:
		spk_rows = df_meta[df_meta.loc[:, 'spk_label'] == id]
		# if id ==0:
		# 	for emt in range(4):
		# 		emt_rows = spk_rows[spk_rows.loc[:, 'emt_label'] == emt]
		# 		chosen_idx += list(np.random.choice(emt_rows.index.values, args.n_emt))
		# else:
		chosen_idx += list(np.random.choice(spk_rows.index.values,args.n_per_spk))

	df_meta_chosen = df_meta.iloc[np.array(sorted(chosen_idx))]

	mel_filenames = [os.path.join(args.input_dir, row.dataset, 'mels', row.mel_filename) for idx,row in df_meta_chosen.iterrows()]
	texts = list(df_meta_chosen.text)

	synth = Synthesizer()
	synth.load(args, checkpoint_path, hparams)
	print("getting embedding for {} samples".format(len(mel_filenames)))
	emb_emt, emb_spk, emb_mo_emt, emb_mo_spk, emb_cont_emt = synth.synthesize(texts, None, None, None, mel_filenames,
																											 mel_ref_filenames_emt=mel_filenames,
																											 mel_ref_filenames_spk=mel_filenames,
																											 emb_only=True)

	#SAVE META + EMBEDDING CSVS
	columns_to_keep = ['dataset', 'mel_filename', 'mel_frames', 'emt_label', 'spk_label', 'basename', 'sex']
	df = df_meta_chosen.loc[:,columns_to_keep]
	df['real'] = 'real'
	df_synth = df.copy()
	df_synth['real'] = 'synth'
	df = pd.concat([df,df_synth])
	df.to_csv(meta_path, sep='\t', index=False)

	# if args.emt_attn:

	# emb_emt = np.vstack((emb_emt, emb_mo_emt))
	emb_spk = np.vstack((emb_spk, emb_mo_spk))


	# pd.DataFrame(emb_emt).to_csv(emb_emt_path,sep='\t',index=False, header=False)
	pd.DataFrame(emb_spk).to_csv(emb_spk_path, sep='\t', index=False, header=False)

	print(len(emb_emt))
	print(emb_emt.shape)

def tacotron_synthesize(args, hparams, checkpoint, sentences=None, model_suffix=None):
	# output_dir = 'tacotron_' + args.output_dir
	output_dir = args.output_dir
	try:
		checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
		log('loaded model at {}'.format(checkpoint_path))
	except:
		raise RuntimeError('Failed to load checkpoint at {}'.format(checkpoint))

	if hparams.tacotron_synthesis_batch_size < hparams.tacotron_num_gpus:
		raise ValueError('Defined synthesis batch size {} is smaller than minimum required {} (num_gpus)! Please verify your synthesis batch size choice.'.format(
			hparams.tacotron_synthesis_batch_size, hparams.tacotron_num_gpus))

	if hparams.tacotron_synthesis_batch_size % hparams.tacotron_num_gpus != 0:
		raise ValueError('Defined synthesis batch size {} is not a multiple of {} (num_gpus)! Please verify your synthesis batch size choice!'.format(
			hparams.tacotron_synthesis_batch_size, hparams.tacotron_num_gpus))

	if args.mode == 'eval':
		return run_eval(args, checkpoint_path, output_dir, hparams, sentences)
	elif args.mode == 'synthesis':
		# return run_synthesis(args, checkpoint_path, output_dir, hparams)
		return run_synthesis_sytle_transfer(args, args.metadata_filename, checkpoint_path, output_dir, hparams)
	elif args.mode == 'synthesis_multiple':
		run_synthesis_multiple(args, checkpoint_path, output_dir, hparams, model_suffix)
	elif args.mode == 'synthesis_random':
		synthesize_random(args, checkpoint_path, output_dir, hparams, model_suffix)
	elif args.mode == 'style_embs':
		get_style_embeddings(args, checkpoint_path, output_dir, hparams)
	else:
		run_live(args, checkpoint_path, hparams)

def test():

	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', default='training_data', help='folder to contain inputs sentences/targets')
	parser.add_argument('--output_dir', default='output', help='folder to contain synthesized mel spectrograms')
	parser.add_argument('--mode', default='synthesis', help='mode for synthesis of tacotron after training')
	parser.add_argument('--GTA', default='False', help='Ground truth aligned synthesis, defaults to True, only considered in Tacotron synthesis mode')
	parser.add_argument('--checkpoint', default=None, help='vary the emotion, speaker id, or neither')
	parser.add_argument('--intercross', action='store_true', default=False, help='whether to use intercross training')
	parser.add_argument('--recon_emb_loss', action='store_true', default=False,
											help='Adds loss for reconstructing embeddings')
	parser.add_argument('--remove_long_samps', action='store_true', default=False, help='removes long samples')
	args = parser.parse_args()

	#set manually
	datasets = 'emt4_jessa' #'vc
	# tk_accent' 'emt4_jessa'
	# suffix = 'emt4'
	model_suffix = 'ej_ae_emb_disc_adv_retrain_ng' #'emt4_jessa_baseline_2'#ej_ae_emb_disc_adv' #mh_emt_disc_l2'
	args.mode =  'synthesis_random' #'synthesis_multiple' #'style_embs' #'synthesis' #'synthesis_random'
	args.paired = False#True
	args.zo = False#True

	#MODEL SETTINGS
	concat = True
	args.emt_only = False#True
	args.remove_long_samps = True#True
	args.tfr_up_only = False
	args.synth_constraint=True
	args.adv_emb_disc = False

	#Other Settings
	args.emt_attn=False#True
	args.emt_ref_gru= 'none' #none' 'gru' 'gru_multi'
	args.attn = 'style_tokens' #'simple' 'multihead'
	args.adain = False#True
	args.flip_spk_emt = False#False

	#EMBEDDING SETTINGS
	args.n_spk = 5
	args.n_emt = 10
	args.n_per_spk = 10

	cur_dir = os.getcwd()
	one_up_dir = os.path.dirname(cur_dir)

	args.intercross = True
	args.GTA=False
	args.nat_gan=False
	args.unpaired=False
	args.input_dir = os.path.join(one_up_dir,'data')
	args.output_dir = os.path.join(one_up_dir,'eval')
	args.metadata_filename = os.path.join(one_up_dir, 'eval/eval_test_emt_attn.txt')#emt_attn.txt')
	# args.train_filename = os.path.join(one_up_dir, 'data/train_{}{}.txt'.format(datasets,suffix))
	args.train_filename = os.path.join(one_up_dir, 'data/train_{}.txt'.format(datasets))
	hparams.tacotron_gst_concat = concat
	args.checkpoint = os.path.join(one_up_dir,'logs/logs-{}/taco_pretrained'.format(model_suffix))

	import socket
	if socket.gethostname() == 'A3907623':
		hparams.tacotron_num_gpus = 1
		hparams.tacotron_batch_size = 32

	tacotron_synthesize(args, hparams, args.checkpoint, model_suffix=model_suffix)


if __name__ == '__main__':
	test()
