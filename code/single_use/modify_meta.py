import os
import numpy as np

if __name__ == '__main__':
	import sys
	sys.path.append(os.getcwd())

from datasets import audio
from hparams import hparams

def add_dataset_name_to_meta(dataset):

	folder_data = os.path.join(os.path.dirname(os.getcwd()), 'data',dataset)

	with open(os.path.join(folder_data,'train.txt'), encoding='utf-8') as f_in:
		with open(os.path.join(folder_data, 'train_new.txt'),'w', encoding='utf-8') as f_out:
			for i,line in enumerate(f_in):
				parts = line.strip().split('|')
				parts = [dataset] + parts
				if dataset == 'emt4':
					parts+=['F']
				f_out.write('|'.join([str(x) for x in parts]) + '\n')

	print("wrote",i,"new lines")

def modify_meta_spk_labels():
	folder_data = os.path.join(os.path.dirname(os.getcwd()), 'data')
	dataset = 'librispeech'

	df_metadata_path = os.path.join(folder_data, 'metadata_{}.txt'.format(dataset))
	with open(df_metadata_path, encoding='utf-8') as f:
		spk_ids = [line.strip().split('|')[3] for line in f]

	spk_ids = sorted(list(frozenset(spk_ids)))

	path_exist_train_folder = r'C:\Users\t-mawhit\Documents\code\Tacotron-2\data\librispeech'

	with open(os.path.join(path_exist_train_folder,'train.txt'), encoding='utf-8') as f_in:
		with open(os.path.join(path_exist_train_folder, 'train_new.txt'),'w', encoding='utf-8') as f_out:
			for i,line in enumerate(f_in):
				parts = line.strip().split('|')
				parts[9] = spk_ids.index(parts[9])+1 #reserving emt4 as spk label 0
				f_out.write('|'.join([str(x) for x in parts]) + '\n')

	print("wrote",i,"new lines")

def re_save_all(wav_path, audio_filename, mel_filename, linear_filename):

	try:
		# Load the audio as numpy array
		aud = audio.load_audio(wav_path, sr=hparams.sample_rate)
	except FileNotFoundError: #catch missing wav exception
		print('file {} present in csv metadata is not present in wav folder. skipping!'.format(wav_path))
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
			raise RuntimeError('audio has invalid value: {}'.format(wav_path))
		if (preem_aud > 1.).any() or (preem_aud < -1.).any():
			raise RuntimeError('audio has invalid value: {}'.format(wav_path))


	#[-1, 1]
	out = aud
	constant_values = 0.
	out_dtype = np.float32

	# Compute the mel scale spectrogram from the audio
	mel_spectrogram = audio.melspectrogram(preem_aud, hparams).astype(np.float32)
	mel_frames = mel_spectrogram.shape[1]

	#Compute the linear scale spectrogram from the audui
	linear_spectrogram = audio.linearspectrogram(preem_aud, hparams).astype(np.float32)
	linear_frames = linear_spectrogram.shape[1]

	#sanity check
	assert linear_frames == mel_frames

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

	# Write the spectrogram and audio to disk
	np.save(audio_filename, out.astype(out_dtype), allow_pickle=False)
	np.save(mel_filename, mel_spectrogram.T, allow_pickle=False)
	np.save(linear_filename, linear_spectrogram.T, allow_pickle=False)

def re_try_mels():
	folder_raw = '/hdfs/intvc/t-mawhit/data/LibriSpeech/train-clean-100'
	folder_data = '/hdfs/intvc/t-mawhit/Tacotron-2/data/librispeech/'
	# folder_data = os.path.join(os.path.dirname(os.getcwd()), 'data',dataset)

	df_train_txt_path = os.path.join(folder_data, 'train.txt')
	with open(df_train_txt_path, encoding='utf-8') as f:
		parts = [line.strip().split('|') for line in f]
		for i,p in enumerate(parts[:]):
			audio_path = os.path.join(folder_data,'audio',p[0])
			mel_path = os.path.join(folder_data,'mels',p[1])
			linear_path = os.path.join(folder_data,'linear',p[2])
			try:
				a = np.load(audio_path)
				# print("loaded audio")
				b = np.load(mel_path)
				# print("loaded mel")
				c = np.load(linear_path)
				# print("loaded linear")
			except:
				print("couldn't load", audio_path,"- now resaving.")
				wav_filename = p[9]
				spk_id, book_id, _ = wav_filename.split('-')
				wav_path = os.path.join(folder_raw,spk_id,book_id,wav_filename)
				re_save_all(wav_path, audio_path,mel_path,linear_path)
				# break
				continue
			if i%1000==0:
				print("completed", i, "samples")
			# break
			# print(audio_path,"worked!")


if __name__ == '__main__':
	modify_meta_spk_labels()
	# re_try_mels()
	# add_dataset_name_to_meta('librispeech')