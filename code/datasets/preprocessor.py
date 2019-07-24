import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from datasets import audio
from wavenet_vocoder.util import is_mulaw, is_mulaw_quantize, mulaw, mulaw_quantize

folder_data = os.path.join(os.path.dirname(os.getcwd()), 'data')

def build_from_path(hparams, args, in_dir, mel_dir, linear_dir, audio_dir, spk_emb_dir, n_jobs=12, tqdm=lambda x: x):
	"""
	Preprocesses the speech dataset from a gven input path to given output directories

	Args:
		- hparams: hyper parameters
		- input_dir: input directory that contains the files to prerocess
		- mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
		- linear_dir: output directory of the preprocessed speech linear-spectrogram dataset
		- wav_dir: output directory of the preprocessed speech audio dataset
		- n_jobs: Optional, number of worker process to parallelize across
		- tqdm: Optional, provides a nice progress bar

	Returns:
		- A list of tuple describing the train examples. this should be written to train.txt
	"""

	# We use ProcessPoolExecutor to parallelize across processes, this is just for
	# optimization purposes and it can be omited
	if not(args.philly):
		executor = ProcessPoolExecutor(max_workers=n_jobs)
	futures = []
	index = 0

	df_metadata_path = os.path.join(folder_data, 'metadata_{}.txt'.format(args.dataset))
	with open(df_metadata_path, encoding='utf-8') as f:
		spk_ids = [line.strip().split('|')[3] for line in f]

	spk_ids = sorted(list(frozenset(spk_ids)))

	with open(df_metadata_path, encoding='utf-8') as f:
		for line in f:
			parts = line.strip().split('|')
			path = parts[0]
			audio_path = os.path.join(in_dir, path)
			text = parts[1]
			emt_label = parts[2]
			spk_label = spk_ids.index(parts[3])+1 #reserving emt4 as spk label 0
			sex = parts[4]
			if args.philly:
				futures.append(_process_utterance(mel_dir, linear_dir, audio_dir, spk_emb_dir, index, audio_path, text, emt_label, spk_label, sex, hparams))
			else:
				futures.append(executor.submit(partial(_process_utterance, mel_dir, linear_dir, audio_dir, spk_emb_dir, index, audio_path, text, emt_label, spk_label, sex, hparams)))
			index += 1

			if args.philly and index % 100 ==0:
				print("samples done:", index)

			#Break after one sample if testing
			if args.TEST:
				break
	if args.philly:
		return [f for f in futures if f is not None]
	else:
		return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_utterance(mel_dir, linear_dir, audio_dir, spk_emb_dir, index, audio_path, text, emt_label, spk_label, sex, hparams):
	"""
	Preprocesses a single utterance wav/text pair

	this writes the mel scale spectogram to disk and return a tuple to write
	to the train.txt file

	Args:
		- mel_dir: the directory to write the mel spectograms into
		- linear_dir: the directory to write the linear spectrograms into
		- wav_dir: the directory to write the preprocessed wav into
		- index: the numeric index to use in the spectogram filename
		- wav_path: path to the audio file containing the speech input
		- text: text spoken in the input audio file
		- hparams: hyper parameters

	Returns:
		- A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, linear_frames, text)
	"""
	try:
		# Load the audio as numpy array
		aud = audio.load_audio(audio_path, sr=hparams.sample_rate)
	except FileNotFoundError: #catch missing wav exception
		print('file {} present in csv metadata is not present in wav folder. skipping!'.format(audio_path))
		return None
	#Trim lead/trail silences
	if hparams.trim_silence:
		aud = audio.trim_silence(aud, hparams)

	#Pre-emphasize
	preem_aud = audio.preemphasis(aud, hparams.preemphasis, hparams.preemphasize)

	#rescale audio
	if hparams.rescale:
		aud = aud / np.abs(aud).max() * hparams.rescaling_max
		preem_aud = preem_aud / np.abs(preem_aud).max() * hparams.rescaling_max

		#Assert all audio is in [-1, 1]
		if (aud > 1.).any() or (aud < -1.).any():
			raise RuntimeError('audio has invalid value: {}'.format(audio_path))
		if (preem_aud > 1.).any() or (preem_aud < -1.).any():
			raise RuntimeError('audio has invalid value: {}'.format(audio_path))

	#Mu-law quantize
	if is_mulaw_quantize(hparams.input_type):
		#[0, quantize_channels)
		out = mulaw_quantize(aud, hparams.quantize_channels)

		#Trim silences
		start, end = audio.start_and_end_indices(out, hparams.silence_threshold)
		aud = aud[start: end]
		preem_aud = preem_aud[start: end]
		out = out[start: end]

		constant_values = mulaw_quantize(0, hparams.quantize_channels)
		out_dtype = np.int16

	elif is_mulaw(hparams.input_type):
		#[-1, 1]
		out = mulaw(aud, hparams.quantize_channels)
		constant_values = mulaw(0., hparams.quantize_channels)
		out_dtype = np.float32

	else:
		#[-1, 1]
		out = aud
		constant_values = 0.
		out_dtype = np.float32

	# Compute the mel scale spectrogram from the audio
	mel_spectrogram = audio.melspectrogram(preem_aud, hparams).astype(np.float32)
	mel_frames = mel_spectrogram.shape[1]

	# if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
	# 	return None

	#Compute the linear scale spectrogram from the audui
	linear_spectrogram = audio.linearspectrogram(preem_aud, hparams).astype(np.float32)
	linear_frames = linear_spectrogram.shape[1]

	#sanity check
	assert linear_frames == mel_frames

	if hparams.use_lws:
		#Ensure time resolution adjustement between audio and mel-spectrogram
		fft_size = hparams.n_fft if hparams.win_size is None else hparams.win_size
		l, r = audio.pad_lr(aud, fft_size, audio.get_hop_size(hparams))

		#Zero pad audio signal
		out = np.pad(out, (l, r), mode='constant', constant_values=constant_values)
	else:
		#Ensure time resolution adjustement between audio and mel-spectrogram
		l_pad, r_pad = audio.librosa_pad_lr(aud, hparams.n_fft, audio.get_hop_size(hparams), hparams.wavenet_pad_sides)

		#Reflect pad audio signal on the right (Just like it's done in Librosa to avoid frame inconsistency)
		out = np.pad(out, (l_pad, r_pad), mode='constant', constant_values=constant_values)

	assert len(out) >= mel_frames * audio.get_hop_size(hparams)

	#time resolution adjustement
	#ensure length of raw audio is multiple of hop size so that we can use
	#transposed convolution to upsample
	out = out[:mel_frames * audio.get_hop_size(hparams)]
	assert len(out) % audio.get_hop_size(hparams) == 0
	time_steps = len(out)

	#Get speaker embedding
	#spk_emb = scoring.get_embedding(spk_emb_model, spk_emb_buckets, audio_path)

	# Write the spectrogram and audio to disk
	audio_filename = 'audio-{}.npy'.format(index)
	mel_filename = 'mel-{}.npy'.format(index)
	linear_filename = 'linear-{}.npy'.format(index)
	spk_emb_filename = 'spkemb-{}.npy'.format(index)
	np.save(os.path.join(audio_dir, audio_filename), out.astype(out_dtype), allow_pickle=False)
	np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
	np.save(os.path.join(linear_dir, linear_filename), linear_spectrogram.T, allow_pickle=False)
	#np.save(os.path.join(spk_emb_dir, spk_emb_filename), spk_emb, allow_pickle=False)

	basename = os.path.basename(audio_path)
	# Return a tuple describing this training example
	return (audio_filename, mel_filename, linear_filename, spk_emb_filename, time_steps, mel_frames, text, emt_label, spk_label, basename, sex)
