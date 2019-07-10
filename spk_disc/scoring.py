import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, euclidean, cosine
from glob import glob

from .model import vggvox_model
from .wav_reader import get_fft_spectrum
from .constants import *


def build_buckets(max_sec, step_sec, frame_step):
	buckets = {}
	frames_per_sec = int(1/frame_step)
	end_frame = int(max_sec*frames_per_sec)
	step_frame = int(step_sec*frames_per_sec)
	for i in range(0, end_frame+1, step_frame):
		s = i
		s = np.floor((s-7+2)/2) + 1  # conv1
		s = np.floor((s-3)/2) + 1  # mpool1
		s = np.floor((s-5+2)/2) + 1  # conv2
		s = np.floor((s-3)/2) + 1  # mpool2
		s = np.floor((s-3+2)/1) + 1  # conv3
		s = np.floor((s-3+2)/1) + 1  # conv4
		s = np.floor((s-3+2)/1) + 1  # conv5
		s = np.floor((s-3)/2) + 1  # mpool5
		s = np.floor((s-1)/1) + 1  # fc6
		if s > 0:
			buckets[i] = int(s)
	return buckets

def get_spk_emb_model():
	model = vggvox_model()
	model.load_weights(WEIGHTS_FILE)
	buckets = build_buckets(MAX_SEC, BUCKET_STEP, FRAME_STEP)
	return(model, buckets)

def get_embedding(model, buckets, path):

	fft = get_fft_spectrum(path, buckets)
	emb = np.squeeze(model.predict(fft.reshape(1,*fft.shape,1)))
	return(emb)

def get_embeddings_from_csv():

	list_file = r'data/embedding/file_list.csv'
	emb_file = r'data/embedding/embedding.tsv'
	meta_file = r'data/embedding/metadata.tsv'

	print("Loading model weights from [{}]....".format(WEIGHTS_FILE))
	model = vggvox_model()
	model.load_weights(WEIGHTS_FILE)
	# model.summary()

	print("Processing samples....")
	buckets = build_buckets(MAX_SEC, BUCKET_STEP, FRAME_STEP)
	meta = pd.read_csv(list_file, delimiter=",")
	result = meta.copy()
	result['features'] = result['filename'].apply(lambda x: get_fft_spectrum(x, buckets))
	result['embedding'] = result['features'].apply(lambda x: np.squeeze(model.predict(x.reshape(1,*x.shape,1))))

	#expand list to columns
	embs = result['embedding'].apply(pd.Series)

	#get just filename, not full path
	meta['filename'] = meta['filename'].apply(lambda x: repr(x).split(r'\\')[-1][:-1])

	#print to csv for use with the tensorflow embedding projector - https://projector.tensorflow.org/
	embs.to_csv(emb_file,sep='\t',index=False,header=False)
	meta.to_csv(meta_file, sep='\t',index=False)

def get_embeddings_from_list_file(model, list_file):
	buckets = build_buckets(MAX_SEC, BUCKET_STEP, FRAME_STEP)
	result = pd.read_csv(list_file, delimiter=",")
	result['features'] = result['filename'].apply(lambda x: get_fft_spectrum(x, buckets))
	result['embedding'] = result['features'].apply(lambda x: np.squeeze(model.predict(x.reshape(1,*x.shape,1))))
	return result[['filename','speaker','embedding']]


def get_id_result():
	print("Loading model weights from [{}]....".format(WEIGHTS_FILE))
	model = vggvox_model()
	model.load_weights(WEIGHTS_FILE)
	model.summary()

	print("Processing enroll samples....")
	enroll_result = get_embeddings_from_list_file(model, ENROLL_LIST_FILE, MAX_SEC)
	enroll_embs = np.array([emb.tolist() for emb in enroll_result['embedding']])
	speakers = enroll_result['speaker']

	print("Processing test samples....")
	test_result = get_embeddings_from_list_file(model, TEST_LIST_FILE, MAX_SEC)
	test_embs = np.array([emb.tolist() for emb in test_result['embedding']])

	print("Comparing test samples against enroll samples....")
	distances = pd.DataFrame(cdist(test_embs, enroll_embs, metric=COST_METRIC), columns=speakers)

	scores = pd.read_csv(TEST_LIST_FILE, delimiter=",",header=0,names=['test_file','test_speaker'])
	scores = pd.concat([scores, distances],axis=1)
	scores['result'] = scores[speakers].idxmin(axis=1)
	scores['correct'] = (scores['result'] == scores['test_speaker'])*1. # bool to int

	print("Writing outputs to [{}]....".format(RESULT_FILE))
	result_dir = os.path.dirname(RESULT_FILE)
	if not os.path.exists(result_dir):
	    os.makedirs(result_dir)
	with open(RESULT_FILE, 'w') as f:
		scores.to_csv(f, index=False)



if __name__ == '__main__':
	# get_id_result()
	# get_embeddings_from_csv()
	PATH = r'\\vibe15\PublicAll\STCM-101\Zo\Wav\Angry\4200002501-4200003000\4200002501.wav'
	model, buckets = get_spk_emb_model()
	emb = get_embedding(model, buckets, PATH)