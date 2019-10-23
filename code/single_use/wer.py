import numpy as np
import os
import pandas as pd
from shutil import copyfile
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
import pickle
from jiwer import wer


# path_asr_results = r'C:\Users\Administrator\Downloads\wavs_new_model_full_jessa_test_set.txt'
# with open(path_asr_results) as f:
# 	line = f.readline()
# lines = line.split(',')
#
# columns=['text_asr']
# df_asr = pd.DataFrame([], columns=columns)
# prev_file = lines[0]
# for l in lines[1:-1]:
#     row = l.split('wav-')
#     df_asr= df_asr.append(pd.DataFrame([[row[0]]],index=[prev_file],columns=columns))
#     prev_file = 'wav-' + row[1]
# df_asr.append(pd.DataFrame([[lines[-1][0]]],index = prev_file,columns=columns))
# print(df_asr.tail())

def create_df_asr(NEW=True):
    folder_asr = r'C:\Users\Administrator\Documents\code\Tacotron-2\results\asr'
    new_suff = 'new' if NEW else 'bsl'

    file = r'wavs_{}_model_full_jessa_test_set.txt'.format(new_suff)
    df_asr = pd.read_csv(os.path.join(folder_asr,file),sep='.wav,',header=None)
    df_asr.columns = ['filename','text_asr']
    df_asr['basename'] = 'na'
    # df_asr.filename = df_asr.filename.apply(lambda x: x+'.wav')

    folder_taco = r'C:\Users\Administrator\Documents\code\Tacotron-2'
    file = r'full_test_set_all - Copy.txt'
    df_meta = pd.read_csv(os.path.join(folder_taco,file),sep='|')

    # print(df_asr.head())
    for i,row in enumerate(df_asr.iterrows()):
        # print(row)
        fname = row[1][0]
        num = fname.split('_')[0].split('-')[1]
        mel_filename = 'mel-{}.npy'.format(num)
        basename = df_meta[df_meta.mel_filename==mel_filename].iloc[0][10]
        df_asr.iloc[i,2] = basename

    path = os.path.join(folder_asr,'df_asr_{}.pickle'.format(new_suff))
    assert (not(os.path.exists(path)))
    with open(path, "wb") as output_file:
        pickle.dump(df_asr, output_file)

def get_scripts(NEW=True):

    folder_asr = r'C:\Users\Administrator\Documents\code\Tacotron-2\results\asr'
    file = r'metadata_jessa.txt'
    df_meta_j = pd.read_csv(os.path.join(folder_asr,file),sep='|')
    df_meta_j.basename = df_meta_j.basename.apply(lambda x: x.split(r'/')[-1])

    new_suff = 'new' if NEW else 'bsl'
    path = os.path.join(folder_asr,'df_asr_{}.pickle'.format(new_suff))
    with open(path, "rb") as output_file:
        df_asr = pickle.load(output_file)

    df_asr['text_true'] = 'na'

    for i,row, in enumerate(df_asr.iterrows()):
        row_meta_j = df_meta_j[df_meta_j.basename==row[1][2]]
        df_asr.iloc[i,3] = row_meta_j.text.iloc[0]
        if i%250==0:
            print('finished:',i)

    #save CSV
    path = os.path.join(folder_asr,'df_asr_{}_with_true.csv'.format(new_suff))
    assert(not(os.path.exists(path)))
    df_asr.to_csv(path)
    print("saved to csv", path)

    #save pickle
    path = os.path.join(folder_asr,'df_asr_{}_with_true.pickle'.format(new_suff))
    assert(not(os.path.exists(path)))

    with open(os.path.join(path), "wb") as output_file:
        pickle.dump(df_asr, output_file)

    print("saved to pickle", path)

def calc_wer(NEW=True):

    folder_asr = r'C:\Users\Administrator\Documents\code\Tacotron-2\results\asr'
    file = r'metadata_jessa.txt'

    new_suff = 'new' if NEW else 'bsl'
    path = os.path.join(folder_asr, 'df_asr_{}_with_true.pickle'.format(new_suff))
    with open(path, "rb") as output_file:
        df_asr = pickle.load(output_file)

    num_rows = len(df_asr.index)
    error = 0
    for i in range(num_rows):
        row = df_asr.iloc[i]
        error+=wer(row.text_true, row.text_asr)
        if i%250==0:
            print('finished:',i)

    print('total acc: {0:.2f}%'.format(100*error/num_rows))
    # time.sleep(.5)
    # break

if __name__ == "__main__":
    NEW=False
    # create_df_asr(NEW=NEW)
    # get_scripts(NEW=NEW)
    calc_wer(NEW=NEW)