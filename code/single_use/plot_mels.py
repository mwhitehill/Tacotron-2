import matplotlib.pyplot as plt
import numpy as np
import os
import librosa
from scipy import signal

if __name__ == '__main__':
	import sys
	sys.path.append(os.getcwd())
from datasets import audio
from hparams import hparams

def plot_mels():
	# mel_folder = r'C:\Users\t-mawhit\Documents\code\Tacotron-2\data\emt4\mels'
	# mel_folder = r'C:\Users\t-mawhit\Documents\code\Tacotron-2\data\vctk_test\mels'
	mel_folder = r'C:\Users\t-mawhit\Documents\code\Tacotron-2\data\emth\mels'
	for i in range(1000,1020):
		mel_name = 'mel-{}.npy'.format(i)
		mel = np.load(os.path.join(mel_folder,mel_name))
		plt.figure()
		plt.title(mel_name)
		plt.imshow(mel.T,origin='lower')

	plt.show()

def test_silence():

	# wav_folder = r'C:\Users\t-mawhit\Documents\data\VCTK-Corpus\wav48'
	# fname = 'p225_001'#''p230_296'
	# user = fname.split('_')[0]
	# path = os.path.join(wav_folder,user,fname+'.wav')

	# wav_folder = r'\\vibe15\PublicAll\STCM-101\Zo\Wav\Happy\4000000001-4000000500'
	wav_folder = r'C:\Users\t-mawhit\Documents\data\Harriton\emotion\Wave16kNormalized\0000000001-0000005000'
	# fname = 'p225_001'#''p230_296'
	files = [f for f in os.listdir(wav_folder) if f.endswith('.wav')]
	for fname in files:
		path = os.path.join(wav_folder,fname)
		for db in [10,15,20,30, 40]:

			aud = librosa.core.load(path, sr=16000)[0]
			# db = 10
			aud = librosa.effects.trim(aud, top_db=db, frame_length=2048, hop_length=512)[0]
			preem_aud = signal.lfilter([1, -.97], [1], aud)
			preem_aud = preem_aud / np.abs(preem_aud).max() * .999

			# librosa.stft(y=preem_aud, n_fft=2048, hop_length=200, win_length=800,pad_mode='constant')

			mel = audio.melspectrogram(preem_aud, hparams).astype(np.float32)
			# a= np.zeros((5,40))
			# mel_frames = mel_spectrogram.shape[1]
			plt.figure()
			plt.title(fname+' ' + '{} db'.format(db))
			plt.imshow(mel, origin='lower')
		plt.show()

def plot_mels_test_eval():
	for i in range(1):
		a = np.load('../eval/mels_save/{}_mel.npy'.format(i))
		b = np.load('../eval/mels_save/{}_ref_emt.npy'.format(i))
		c = np.load('../eval/mels_save/{}_ref_spk.npy'.format(i))
		plt.figure()
		plt.title(str(i) + '_' + 'mel')
		plt.imshow(a[:500,:].T,origin='lower')
		plt.figure()
		plt.title(str(i) + '_' + 'emt')
		plt.imshow(b[:500, :].T, origin='lower')
		plt.figure()
		plt.title(str(i) + '_' + 'spk')
		plt.imshow(c[:500, :].T, origin='lower')


	plt.show()

def plot_mels_silence():

	folder_data = os.path.join(os.path.dirname(os.getcwd()), 'data')

	dataset='vctk'
	out_dir = os.path.join(folder_data, dataset + '_test_silence')
	# out_dir = os.path.join(folder_data, dataset)
	out_dir = r'C:\Users\t-mawhit\Documents\code\Tacotron-2\data\vctk_test'

	mel_dir = os.path.join(out_dir, 'mels')
	index=0

	plt.figure()
	a = np.load(os.path.join(mel_dir, 'mel-{}.npy'.format(index)))
	plt.imshow(a.T, origin='lower')
	plt.show()

def plot_mels_unpaired():

	folder = r'C:\Users\t-mawhit\Documents\code\Tacotron-2\eval\mels_save'
	mel_folder = os.path.join(folder,'mels')
	wav_folder = os.path.join(folder,'wavs')
	os.makedirs(mel_folder, exist_ok=True)
	os.makedirs(wav_folder, exist_ok=True)
	# for i in range(32):
	# 	mel_name = 'mel_out_{}.npy'.format(i)
	# 	mel = np.load(os.path.join(mel_folder,mel_name))
	# 	plt.figure()
	# 	plt.title(mel_name)
	# 	plt.imshow(mel.T, origin='lower')

	for i in range(32):
		mel_name = '{}_up.npy'.format(i)
		mel = np.load(os.path.join(folder,mel_name))
		plt.figure()
		plt.title(mel_name)
		plt.imshow(mel.T, origin='lower')

	# for i in range(32):
	# 	mel_name = 'mel_out_unp_{}.npy'.format(i)
	# 	mel = np.load(os.path.join(mel_folder,mel_name))
	#
	# 	wav = audio.inv_mel_spectrogram(mel.T, hparams)
	# 	audio.save_wav(wav, os.path.join(wav_folder, mel_name.split('.')[0]+'.wav'), sr=hparams.sample_rate)

	plt.show()

if __name__ == '__main__':
	# plot_mels()
	# plot_mels_silence()
	# test_silence()
	plot_mels_unpaired()