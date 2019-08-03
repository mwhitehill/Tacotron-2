import argparse
import os
import re
import time
from time import sleep
import numpy as np
import tensorflow as tf
import pandas as pd

if __name__ == '__main__':
	import sys
	sys.path.append(os.getcwd())
	import argparse

from hparams import hparams, hparams_debug_string
from infolog import log
from tacotron.synthesizer import Synthesizer
from tqdm import tqdm
from tacotron.feeder import get_metadata_df


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

def run_synthesis_sytle_transfer(args, checkpoint_path, output_dir, hparams):
	GTA = (args.GTA == 'True')
	if GTA:
		synth_dir = os.path.join(output_dir, 'gta')

		#Create output path if it doesn't exist
		os.makedirs(synth_dir, exist_ok=True)
	else:
		synth_dir = os.path.join(output_dir, 'natural')

		#Create output path if it doesn't exist
		os.makedirs(synth_dir, exist_ok=True)

	log(hparams_debug_string())
	synth = Synthesizer()
	synth.load(checkpoint_path, hparams, gta=GTA)
	with open(args.metadata_filename, encoding='utf-8') as f:
		metadata = [line.strip().split('|') for line in f if not(line.startswith('#'))]
		frame_shift_ms = hparams.hop_size / hparams.sample_rate
		hours = sum([int(x[6]) for x in metadata]) * frame_shift_ms / (3600)
		log('Loaded metadata for {} examples ({:.2f} hours)'.format(len(metadata), hours))

	log('Starting Synthesis')
	texts = [m[7] for m in metadata]
	mel_filenames = [os.path.join(args.input_dir, m[0], 'mels', m[2]) for m in metadata]
	basenames = [os.path.basename(m).replace('.npy', '').replace('mel-', '') for m in mel_filenames]
	basenames_refs = [m[11]+'_'+m[13] for m in metadata]

	mel_ref_filenames_emt = []
	mel_ref_filenames_spk = []
	for m in metadata:
			if m[12] == 'same':
				mel_ref_filenames_emt.append(os.path.join(args.input_dir, m[0], 'mels', m[2]))
			else:
				mel_ref_filenames_emt.append(os.path.join(args.input_dir, 'emt4', 'mels', m[12]))

			if m[14] == 'same':
				mel_ref_filenames_spk.append(os.path.join(args.input_dir, m[0],'mels', m[2]))
			else:
				mel_ref_filenames_spk.append(os.path.join(args.input_dir, 'librispeech', 'mels', m[14]))

	mel_output_filenames, speaker_ids = synth.synthesize(texts, basenames, synth_dir, synth_dir, mel_filenames,
																											 basenames_refs=basenames_refs,
																											 mel_ref_filenames_emt=mel_ref_filenames_emt,
																											 mel_ref_filenames_spk=mel_ref_filenames_spk)
	return None

def get_style_embeddings(args, checkpoint_path, output_dir, hparams):

	emb_dir = os.path.join(output_dir, 'embeddings')
	os.makedirs(emb_dir, exist_ok=True)
	meta_path = os.path.join(emb_dir,'meta.tsv')
	emb_emt_path = os.path.join(emb_dir,'emb_emt.tsv')
	emb_spk_path = os.path.join(emb_dir,'emb_spk.tsv')

	with open(args.train_filename , encoding='utf-8') as f:
		metadata = [line.strip().split('|') for line in f if not(line.startswith('#'))]

	df_meta = get_metadata_df(args.train_filename)

	spk_ids = df_meta.spk_label.unique()
	spk_ids_chosen = np.sort(np.random.choice(spk_ids,args.n_spk))

	#make sure first user is in embeddings (zo - the one with emotions)
	if not(0 in spk_ids_chosen):
		spk_ids_chosen = np.sort(np.append(spk_ids_chosen,0))

	chosen_idx = []
	for id in spk_ids_chosen:
		spk_rows = df_meta[df_meta.loc[:, 'spk_label'] == id]
		if id ==0:
			for emt in range(4):
				emt_rows = spk_rows[spk_rows.loc[:, 'emt_label'] == emt]
				chosen_idx += list(np.random.choice(emt_rows.index.values, args.n_emt))
		else:
			chosen_idx += list(np.random.choice(spk_rows.index.values,args.n_per_spk))

	df_meta_chosen = df_meta.iloc[np.array(sorted(chosen_idx))]

	mel_filenames = [os.path.join(args.input_dir, row.dataset, 'mels', row.mel_filename) for idx,row in df_meta_chosen.iterrows()]
	texts = list(df_meta_chosen.text)

	synth = Synthesizer()
	synth.load(checkpoint_path, hparams)
	print("getting embedding for {} samples".format(len(mel_filenames)))

	emb_emt, emb_spk = synth.synthesize(texts, None, None, None, mel_filenames,
																											 mel_ref_filenames_emt=mel_filenames,
																											 mel_ref_filenames_spk=mel_filenames,
																											 emb_only=True)

	#SAVE META + EMBEDDING CSVS
	columns_to_keep = ['dataset', 'mel_filename', 'mel_frames', 'emt_label', 'spk_label', 'basename', 'sex']
	df_meta_chosen.loc[:,columns_to_keep].to_csv(meta_path, sep='\t', index=False)

	pd.DataFrame(emb_emt).to_csv(emb_emt_path,sep='\t',index=False, header=False)
	pd.DataFrame(emb_spk).to_csv(emb_spk_path, sep='\t', index=False, header=False)

def tacotron_synthesize(args, hparams, checkpoint, sentences=None):
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
		return run_synthesis_sytle_transfer(args, checkpoint_path, output_dir, hparams)
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
	args = parser.parse_args()


	#set manually
	model_suffix = '2conds_disc_orthog'
	args.mode = 'style_embs' #'synthesis'

	#MODEL SETTINGS
	concat = True

	#EMBEDDING SETTINGS
	args.n_spk = 5
	args.n_emt = 10
	args.n_per_spk = 10

	cur_dir = os.getcwd()
	one_up_dir = os.path.dirname(cur_dir)

	args.intercross = True
	args.GTA=False
	args.input_dir = os.path.join(one_up_dir,'data')
	args.output_dir = os.path.join(one_up_dir,'eval')
	args.metadata_filename = os.path.join(one_up_dir, 'eval/eval_test.txt')
	args.train_filename = os.path.join(one_up_dir, 'data/train.txt')
	hparams.tacotron_gst_concat = concat
	args.checkpoint = os.path.join(one_up_dir,'logs/logs-Tacotron-2_{}/taco_pretrained'.format(model_suffix))

	tacotron_synthesize(args, hparams, args.checkpoint)


if __name__ == '__main__':
	test()
