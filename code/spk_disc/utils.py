import tensorflow as tf
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
import random
from configuration import get_config
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    import argparse

import sys
sys.path.append(os.getcwd())
from hparams import hparams

config = get_config()
TEST_SIZE =.05
columns_to_keep = ['dataset', 'mel_filename', 'mel_frames', 'emt_label', 'spk_label', 'basename', 'sex']

def keyword_spot(spec):
    """ Keyword detection for data preprocess
        For VTCK data I truncate last 80 frames of trimmed audio - "Call Stella"
    :return: 80 frames spectrogram
    """
    return spec[:, -config.tdsv_frame:]


def random_batch_old(speaker_num=4,#config.N,
                     utter_num=5, #config.M,
                     shuffle=True, noise_filenum=None, utter_start=0,
                     TEST=False):
    """ Generate 1 batch.
        For TD-SV, noise is added to each utterance.
        For TI-SV, random frame length is applied to each batch of utterances (140-180 frames)
        speaker_num : number of speaker of each batch
        utter_num : number of utterance per speaker of each batch
        shuffle : random sampling or not
        noise_filenum : specify noise file or not (TD-SV)
        utter_start : start point of slicing (TI-SV)
    :return: 1 random numpy batch (frames x batch(NM) x n_mels)
    """

    # data path
    if TEST:
        path = config.test_path
    else:
        path = config.train_path

    # TD-SV
    # if config.tdsv:
    #     np_file = os.listdir(path)[0]
    #     path = os.path.join(path, np_file)  # path of numpy file
    #     utters = np.load(path)              # load text specific utterance spectrogram
    #     if shuffle:
    #         np.random.shuffle(utters)       # shuffle for random sampling
    #     utters = utters[:speaker_num]       # select N speaker
    #
    #     # concat utterances (M utters per each speaker)
    #     # ex) M=2, N=2 => utter_batch = [speaker1, speaker1, speaker2, speaker2]
    #     utter_batch = np.concatenate([np.concatenate([utters[i]]*utter_num, axis=1) for i in range(speaker_num)], axis=1)
    #
    #     if noise_filenum is None:
    #         noise_filenum = np.random.randint(0, config.noise_filenum)                    # random selection of noise
    #     noise = np.load(os.path.join(config.noise_path, "noise_%d.npy"%noise_filenum))  # load noise
    #
    #     utter_batch += noise[:,:utter_batch.shape[1]]   # add noise to utterance
    #
    #     utter_batch = np.abs(utter_batch) ** 2
    #     mel_basis = librosa.filters.mel(sr=config.sr, n_fft=config.nfft, n_mels=40)
    #     utter_batch = np.log10(np.dot(mel_basis, utter_batch) + 1e-6)          # log mel spectrogram of utterances
    #
    #     utter_batch = np.array([utter_batch[:,config.tdsv_frame*i:config.tdsv_frame*(i+1)]
    #                             for i in range(speaker_num*utter_num)])        # reshape [batch, n_mels, frames]
    #
    # # TI-SV
    # else:
    np_file_list = [f for f in os.listdir(path) if f.endswith('.npy')]
    total_speaker = len(np_file_list)

    if shuffle:
        selected_files = random.sample(np_file_list, speaker_num)  # select random N speakers
    else:
        selected_files = np_file_list[:speaker_num]                # select first N speakers

    utter_batch = []
    for file in selected_files:
        utters = np.load(os.path.join(path, file))        # load utterance spectrogram of selected speaker
        if shuffle:
            utter_index = np.random.randint(0, utters.shape[0], utter_num)   # select M utterances per speaker
            utter_batch.append(utters[utter_index])       # each speakers utterance [M, n_mels, frames] is appended
        else:
            utter_batch.append(utters[utter_start: utter_start+utter_num])

    utter_batch = np.concatenate(utter_batch, axis=0)     # utterance batch [batch(NM), n_mels, frames]

    #keep all at fixed size of 140 frames
    # if config.train:
    #     frame_slice = np.random.randint(140,181)          # for train session, random slicing of input batch
    #     utter_batch = utter_batch[:,:,:frame_slice]
    # else:
    #     utter_batch = utter_batch[:,:,:160]               # for train session, fixed length slicing of input batch

    utter_batch = np.transpose(utter_batch, axes=(2,0,1))     # transpose [frames, batch, n_mels]

    return utter_batch


def normalize(x):
    """ normalize the last dimension vector of the input matrix
    :return: normalized input
    """
    return x/tf.sqrt(tf.reduce_sum(x**2, axis=-1, keep_dims=True)+1e-6)


def cossim(x,y, normalized=True):
    """ calculate similarity between tensors
    :return: cos similarity tf op node
    """
    if normalized:
        return tf.reduce_sum(x*y)
    else:
        x_norm = tf.sqrt(tf.reduce_sum(x**2)+1e-6)
        y_norm = tf.sqrt(tf.reduce_sum(y**2)+1e-6)
        return tf.reduce_sum(x*y)/x_norm/y_norm


def similarity(embedded, w, b, N, M, P=config.proj, center=None):
    """ Calculate similarity matrix from embedded utterance batch (NM x embed_dim) eq. (9)
        Input center to test enrollment. (embedded for verification)
    :return: tf similarity matrix (NM x N)
    """
    embedded_split = tf.reshape(embedded, shape=[N, M, P])

    if center is None:
        center = normalize(tf.reduce_mean(embedded_split, axis=1))              # [N,P] normalized center vectors eq.(1)
        center_except = normalize(tf.reshape(tf.reduce_sum(embedded_split, axis=1, keep_dims=True) - embedded_split, shape=[N*M,P]))  # [NM,P] center vectors eq.(8)
        # make similarity matrix eq.(9)
        S = tf.concat(
            [tf.concat([tf.reduce_sum(center_except[i*M:(i+1)*M,:]*embedded_split[j,:,:], axis=1, keep_dims=True) if i==j
                        else tf.reduce_sum(center[i:(i+1),:]*embedded_split[j,:,:], axis=1, keep_dims=True) for i in range(N)],
                       axis=1) for j in range(N)], axis=0)
    else :
        # If center(enrollment) exist, use it.
        S = tf.concat(
            [tf.concat([tf.reduce_sum(center[i:(i + 1), :] * embedded_split[j, :, :], axis=1, keep_dims=True) for i
                        in range(N)],
                       axis=1) for j in range(N)], axis=0)

    S = tf.abs(w)*S+b   # rescaling

    return S


def loss_cal(S, N, M, type="softmax"):
    """ calculate loss with similarity matrix(S) eq.(6) (7) 
    :type: "softmax" or "contrast"
    :return: loss
    """
    S_correct = tf.concat([S[i*M:(i+1)*M, i:(i+1)] for i in range(N)], axis=0)  # colored entries in Fig.1

    if type == "softmax":
        total = -tf.reduce_sum(S_correct-tf.log(tf.reduce_sum(tf.exp(S), axis=1, keep_dims=True) + 1e-6))
    elif type == "contrast":
        S_sig = tf.sigmoid(S)
        S_sig = tf.concat([tf.concat([0*S_sig[i*M:(i+1)*M, j:(j+1)] if i==j
                              else S_sig[i*M:(i+1)*M, j:(j+1)] for j in range(N)], axis=1)
                             for i in range(N)], axis=0)
        total = tf.reduce_sum(1-tf.sigmoid(S_correct)+tf.reduce_max(S_sig, axis=1, keep_dims=True))
    else:
        raise AssertionError("loss type should be softmax or contrast !")

    return total


def optim(lr):
    """ return optimizer determined by configuration
    :return: tf optimizer
    """
    if config.optim == "sgd":
        return tf.train.GradientDescentOptimizer(lr)
    elif config.optim == "rmsprop":
        return tf.train.RMSPropOptimizer(lr)
    elif config.optim == "adam":
        return tf.train.AdamOptimizer(lr)#, beta1=config.beta1, beta2=config.beta2)
    else:
        raise AssertionError("Wrong optimizer type!")




class Feeder:
    """
      Feeds batches of data into queue on a background thread.
    """

    def __init__(self, metadata_filename, args, hparams):
        super(Feeder, self).__init__()

        self.args = args
        self.hparams = hparams

        # Load metadata
        self.data_folder = os.path.dirname(metadata_filename)

        with open(metadata_filename, encoding='utf-8') as f:
            self._metadata = [line.strip().split('|') for line in f]
            if args.remove_long_samps:
                len_before = len(self._metadata)
                # remove the 2 longest utterances + any over 900 frames long
                self._metadata = [f for f in self._metadata if not (f[10].endswith('_023.wav'))]
                self._metadata = [f for f in self._metadata if not (f[10].endswith('_021.wav'))]
                self._metadata = [f for f in self._metadata if int(f[6]) < 500]
                print("Removed Long Samples")
                print("# samps before:", len_before)
                print("# samps after:", len(self._metadata))
            if args.model_type == 'accent':
                len_before = len(self._metadata)
                self._metadata = [f for f in self._metadata if (int(f[8]) in [0,2,3,5,8])]
                print("Kept only 5 largest accents: American, Canadian, English, Irish, Scottish")
                print("# samps before:", len_before)
                print("# samps after:", len(self._metadata))
            frame_shift_ms = config.hop / config.sr
            hours = sum([int(x[6]) for x in self._metadata]) * frame_shift_ms / (3600)
            print('Loaded metadata for {} examples ({:.2f} hours)'.format(len(self._metadata), hours))

        # self._metadata_df = get_metadata_df(metadata_filename, args)
        self._metadata_df = pd.DataFrame(self._metadata)
        columns = ['dataset', 'audio_filename', 'mel_filename', 'linear_filename', 'spk_emb_filename', 'time_steps',
                   'mel_frames', 'text', 'emt_label', 'spk_label', 'basename', 'sex']
        self._metadata_df.columns = columns

        indices = np.arange(len(self._metadata))
        train_indices, test_indices = train_test_split(indices,test_size=TEST_SIZE,
                                                       random_state=self.hparams.tacotron_data_random_state)

        # Make sure test_indices is a multiple of batch_size else round down
        len_test_indices = self._round_down(len(test_indices), args.N * args.M)
        extra_test = test_indices[len_test_indices:]
        test_indices = test_indices[:len_test_indices]
        train_indices = np.concatenate([train_indices, extra_test])

        self._train_meta = list(np.array(self._metadata)[train_indices])
        self._test_meta = list(np.array(self._metadata)[test_indices])

        if args.test_max_len:
            self._train_meta.sort(key=lambda x: int(x[6]), reverse=True)
            self._test_meta.sort(key=lambda x: int(x[6]), reverse=True)
            print("TESTING MAX LENGTH FOR SAMPLES TO FIND MAX BATCH SIZE")

        self._metadata_df['train_test'] = 'train'
        self._metadata_df.iloc[np.array(sorted(test_indices)) - 1, -1] = 'test'
        self.df_meta_train = self._metadata_df[self._metadata_df.loc[:, 'train_test'] == 'train']
        self.df_meta_test = self._metadata_df[self._metadata_df.loc[:, 'train_test'] == 'test']

        self.total_emt = self._metadata_df.emt_label.unique()
        self.total_spk = self._metadata_df.spk_label.unique()

        # num_samps = np.zeros(len(self.total_emt))
        # self.class_weights = np.zeros(len(self.total_emt))
        # if args.model_type == 'emt':
        #     for e in self.total_emt:
        #         print(len(self.df_meta_train.emt_label[self.df_meta_train.emt_label == e].index))
        #         # num_samps[i] = self.df_meta_train.emt_label ==
        #         raise

        if hparams.symmetric_mels:
            self._target_pad = -hparams.max_abs_value
        else:
            self._target_pad = 0.

    def random_batch(self, TEST=False, make_meta=False):
        mels = []
        df = self.df_meta_test if TEST else self.df_meta_train
        if self.args.model_type == 'emt':
            labels = np.random.choice(self.total_emt, self.args.N, replace=False)
            #just use the emotion dataset
            df = df[df['dataset'] == 'emt4']
            # df = df[df['dataset'].isin(['emt4', 'emth'])]
            model_label = 'emt'
        elif self.args.model_type == 'spk':
            #only use general emotion for speaker model
            df = df[df['emt_label'] == '0']
            labels = np.random.choice(self.total_spk, self.args.N, replace=False)
            model_label = 'spk'
        elif self.args.model_type == 'accent':
            labels = np.random.choice(self.total_emt, self.args.N, replace=False)
            model_label = 'emt'
        else:
            raise ValueError('invalid discriminator type - must be emt or spk')
        label_type = '{}_label'.format(model_label)

        idxs_all = []
        labels_all = []
        for label in labels:
            # df_meta_same_style = self.df_meta_train[self.df_meta_train['dataset'].isin(['emt4', 'emth'])]
            df_meta_same_style = df[df[label_type] == label]

            # select M mels from same style to use as reference
            idxs = np.random.choice(df_meta_same_style.index, self.args.M)
            idxs_all += list(idxs)
            for idx in idxs:
                mel_name_same_style = df_meta_same_style.loc[idx, 'mel_filename']
                dataset_same_style = df_meta_same_style.loc[idx, 'dataset']
                mels.append(np.load(os.path.join(self.data_folder, dataset_same_style, 'mels', mel_name_same_style)))
                labels_all.append(int(label))

        mels, max_len = self._prepare_targets(mels, self.hparams.outputs_per_step)
        meta = df.loc[idxs_all, columns_to_keep] if make_meta else None

        return(mels, meta, np.array(labels_all))

    def random_batch_disc(self, TEST=False, make_meta=False):
        mels = []
        df = self.df_meta_test if TEST else self.df_meta_train

        if self.args.model_type == 'emt':
            #just use the emotion dataset
            df = df[df['dataset'] == 'emt4']
            # df = df[df['dataset'].isin(['emt4', 'emth'])]
            model_label = 'emt'
        elif self.args.model_type == 'spk':
            model_label = 'spk'
        elif self.args.model_type == 'accent':
            model_label = 'emt'
        else:
            raise ValueError('invalid discriminator type - must be emt or spk')

        label_type = '{}_label'.format(model_label)
        labels_all = []

        idxs = np.random.choice(df.index,self.args.N*self.args.M,replace=False)

        for idx in idxs:
            mel_name_same_style = df.loc[idx, 'mel_filename']
            dataset_same_style = df.loc[idx, 'dataset']
            mels.append(np.load(os.path.join(self.data_folder, dataset_same_style, 'mels', mel_name_same_style)))
            labels_all.append(int(df.loc[idx, label_type]))

        mels, max_len = self._prepare_targets(mels, self.hparams.outputs_per_step)
        meta = df.loc[idxs, columns_to_keep] if make_meta else None

        return(mels, meta, np.array(labels_all))

    def emb_batch(self, datasets, TEST=False, make_meta=False):
        mels = []
        df = self.df_meta_test if TEST else self.df_meta_train
        if self.args.model_type == 'emt':
            labels = np.random.choice(self.total_emt, self.args.N, replace=False)
            #just use the emotion dataset
            df = df[df['dataset'].isin(datasets)]
        elif self.args.model_type == 'spk':
            labels = np.random.choice(self.total_spk, self.args.N, replace=False)
        else:
            raise ValueError('invalid discriminator type - must be emt or spk')
        label_type = '{}_label'.format(self.args.model_type)

        idxs_all = []
        for dset in datasets:
            if dset =='vctk':
                labels = np.array(['0','0','0','0'])#
            df_dataset = df[df['dataset'] == dset]

            for label in labels:
                # df_meta_same_style = self.df_meta_train[self.df_meta_train['dataset'].isin(['emt4', 'emth'])]
                df_meta_same_style = df_dataset[df_dataset[label_type] == label]

                # select M mels from same style to use as reference
                idxs = np.random.choice(df_meta_same_style.index, self.args.M)
                idxs_all += list(idxs)
                for idx in idxs:
                    mel_name_same_style = df_meta_same_style.loc[idx, 'mel_filename']
                    dataset_same_style = df_meta_same_style.loc[idx, 'dataset']
                    mels.append(np.load(os.path.join(self.data_folder, dataset_same_style, 'mels', mel_name_same_style)))

        mels, max_len = self._prepare_targets(mels, self.hparams.outputs_per_step)
        meta = df.loc[idxs_all, columns_to_keep] if make_meta else None

        return(mels, meta)

    def _prepare_targets(self, targets, alignment):
        max_len = max([len(t) for t in targets])
        data_len = self._round_up(max_len, alignment)
        return np.stack([self._pad_target(t, data_len) for t in targets]), data_len

    def _round_down(self, x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x - remainder

    def _round_up(self, x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x + multiple - remainder

    def _pad_target(self, t, length):
        return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=self._target_pad)

def test_batch(data_path, df, args):

    model_label = 'emt' if args.model_type == 'emt' else 'spk'

    label_type = '{}_label'.format(model_label)
    labels = df.loc[:,label_type].values

    #basenames = df.loc[:, 'basename'].values
    #filenames = [os.path.join(data_path,'mels',b) for b in basenames]

    emt_names = df.loc[:, 'emt_name'].values
    spk_names = df.loc[:, 'spk_name'].values
    mel_numbers = df.loc[:, 'mel_filename'].values
    mel_numbers = [m.split('-')[1].split('.')[0] for m in mel_numbers]
    filenames = [os.path.join(data_path, 'mel-{}_{}_{}.npy'.format(mel_numbers[i], emt_names[i], spk_names[i])) for i in range(len(mel_numbers))]

    mels = [np.load(f) for f in filenames]
    mels, max_len = _prepare_targets(mels, hparams.outputs_per_step)

    meta = df.loc[:, columns_to_keep]

    return(mels, meta, labels)

def _prepare_targets(targets, alignment):
    max_len = max([len(t) for t in targets])
    data_len = _round_up(max_len, alignment)
    return np.stack([_pad_target(t, data_len) for t in targets]), data_len

def _round_down(x, multiple):
    remainder = x % multiple
    return x if remainder == 0 else x - remainder

def _round_up(x, multiple):
    remainder = x % multiple
    return x if remainder == 0 else x + multiple - remainder

def _pad_target(t, length):
    _target_pad = -hparams.max_abs_value if hparams.symmetric_mels else 0.
    return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=_target_pad)



# for check
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=4, help='Number groups')
    parser.add_argument('--M', type=int, default=5, help='Number utterances per group')
    parser.add_argument('--remove_long_samps', action='store_true', default=False,
                        help='Will remove out the longest samples from EMT4/VCTK')
    parser.add_argument('--test_max_len', action='store_true', default=False,
                        help='Will create batches with the longest samples first to test max batch size')
    parser.add_argument('--TEST', action='store_true', default=False,
                        help='Uses small groups of batches to make testing faster')
    parser.add_argument('--train_filename', default='../data/train_emt4_vctk_e40_v15.txt')
    parser.add_argument('--model_type', default='emt', help='Options = emt or spk')
    parser.add_argument('--time_string', default=None, help='time string of previous saved model')
    parser.add_argument('--restore', action='store_true', default=False,
                        help='whether to restore the model')
    args = parser.parse_args()
    args.remove_long_samps = True  # True
    args.test_max_len = False  # True
    args.TEST = True
    args.model_type = 'spk'
    args.N=2

    # feeder = Feeder('../data/train_emt4_jessa.txt', args, hparams)
    # mels, meta, labels = feeder.random_batch_disc(TEST=False, make_meta=True)
    # print(meta)
    DATA_PATH = r'C:\Users\t-mawhit\Documents\code\Tacotron-2\eval\random\emt4_jessa_baseline_2\2019.09.14_18-21-43'
    input_meta_path = os.path.join(DATA_PATH, 'meta.csv')
    df = pd.read_csv(input_meta_path)
    mels, meta, labels = test_batch(DATA_PATH,args)
    # print(mels.shape)
    print(labels)
    # w= tf.constant([1], dtype= tf.float32)
    # b= tf.constant([0], dtype= tf.float32)
    # embedded = tf.constant([[0,1,0],[0,0,1], [0,1,0], [0,1,0], [1,0,0], [1,0,0]], dtype= tf.float32)
    # sim_matrix = similarity(embedded,w,b,3,2,3)
    # loss1 = loss_cal(sim_matrix, type="softmax",N=3,M=2)
    # loss2 = loss_cal(sim_matrix, type="contrast",N=3,M=2)
    # with tf.Session() as sess:
    #     print(sess.run(sim_matrix))
    #     print(sess.run(loss1))
    #     print(sess.run(loss2))