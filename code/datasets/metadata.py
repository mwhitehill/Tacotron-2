import os
import pandas as pd
import numpy as np
import librosa
if __name__ == '__main__':
	import sys
	sys.path.append(os.getcwd())
from datasets import audio

folder_data = os.path.join(os.path.dirname(os.getcwd()), 'data')

def create_metadata_emt4():

  folder_wav = '//vibe15/PublicAll/STCM-101/Zo/Wav'
  folder_data_emt4 = os.path.join(folder_data, 'emt4')
  os.makedirs(folder_data_emt4,exist_ok=True)

  all_txt_path = os.path.join(folder_data, 'all_txt_wav.txt')
  df_all_txt = pd.read_csv(all_txt_path, sep='|', index_col=0,
                   names=['filename', 'script', 'emotion_label'])
  df_all_txt.emotion_label = df_all_txt.emotion_label.apply(int)

  paths = []
  for i,(root, dirs, files) in enumerate(os.walk(folder_wav, topdown=True)):
    for f in files:
      paths.append(os.path.join(os.path.basename(os.path.dirname(root)), os.path.basename(root),f))

  columns = ['path', 'script', 'emt_label', 'spk_id', 'sex']
  df_metadata = pd.DataFrame([],columns=columns)

  for p in paths[:]:
    fname = os.path.basename(p)
    name = int(fname.split('.')[0])
    row = df_all_txt.loc[name]
    script = row.script
    emt_label = row.emotion_label
    df_metadata = df_metadata.append(pd.DataFrame([[p.replace('\\','/'),script,emt_label,0,'F']],columns=columns),ignore_index=True)
  df_metadata_path = os.path.join(folder_data, 'metadata_emt4.txt')
  df_metadata.to_csv(df_metadata_path,sep='|',header=False,index=False)

def create_metadata_jessa():

  folder_wav = '//vibe15/PublicAll/STCM-101/Jessa/wave16kNormalized'
  folder_data_jessa = os.path.join(folder_data, 'jessa')
  os.makedirs(folder_data_jessa,exist_ok=True)

  paths = []
  for i,(root, dirs, files) in enumerate(os.walk(folder_wav, topdown=True)):
    for f in files:
      paths.append(os.path.join(os.path.basename(os.path.dirname(root)), os.path.basename(root),f))

  columns = ['path', 'script', 'emt_label', 'spk_id', 'sex']
  df_metadata = pd.DataFrame([],columns=columns)

  for i,p in enumerate(paths[:]):
    fname = os.path.basename(p)
    name = fname.split('.')[0]
    text_file = os.path.basename(os.path.dirname(p)) + '.txt'
    text_file_path  = os.path.join(os.path.dirname(folder_wav),'TextScripts_UTF8',text_file)
    df = pd.read_csv(text_file_path,sep=r'\t',header=None,names=['filename', 'script'],dtype={'filename': 'object'},index_col=0)
    df.index.values[0] = df.index.values[0][3:]
    row = df.loc[name]
    script = row.script
    if script == None:
      print("script not found", p)
    df_metadata = df_metadata.append(pd.DataFrame([[p.replace('\\','/'),script,0,1,'F']],columns=columns),ignore_index=True)

    if i % 200 == 0:
      print("finished",i)

  df_metadata_path = os.path.join(folder_data, 'metadata_jessa.txt')
  df_metadata.to_csv(df_metadata_path,sep='|',header=False,index=False)

def create_metadata_harriton():

  folder_raw = r'C:\Users\t-mawhit\Documents\data\Harriton\emotion'
  folder_wav = os.path.join(folder_raw, 'Wave16kNormalized')
  folder_data_emth= os.path.join(folder_data, 'emth')
  os.makedirs(folder_data_emth,exist_ok=True)

  all_txt_path = os.path.join(folder_raw, 'all_txt_wav.txt')
  df_all_txt = pd.read_csv(all_txt_path, encoding='utf-8',sep=r'\t',names=['filename', 'script'],dtype={'filename': 'object'},engine='python')

  df_all_txt['emotion_label'] = 0
	
	#swap emotion labels for harriton to match Zo (i.e. angry is labelled 1 in harriton but is 2 in zo)
  for i1,i2 in [('1','2'),('2','1'),('3','3')]:
    idxs = df_all_txt[df_all_txt['filename'].map(lambda x: x.startswith(i1))].index
    df_all_txt.loc[idxs,'emotion_label'] = int(i2)

  paths = []
  for i,(root, dirs, files) in enumerate(os.walk(folder_wav, topdown=True)):
    for f in files:
      if not(f.endswith('.wav') or f.endswith('.flac')):
        print("non audio file -", f)
        continue
      paths.append(os.path.join(os.path.basename(os.path.dirname(root)), os.path.basename(root),f))

  columns = ['path', 'script', 'emt_label', 'spk_id', 'sex']
  df_metadata = pd.DataFrame([],columns=columns)

  for p in paths[:]:
    fname = os.path.basename(p)
    name = fname.split('.')[0]
    row = df_all_txt[df_all_txt.filename == name]
    script = row.script.values[0]
    emt_label = row.emotion_label.values[0]
    df_metadata = df_metadata.append(pd.DataFrame([[p.replace('\\','/'),script,emt_label,1,'M']],columns=columns),ignore_index=True)
  df_metadata_path = os.path.join(folder_data, 'metadata_emth.txt')
  df_metadata.to_csv(df_metadata_path,sep='|',header=False,index=False)


def get_librispeech_transcript(root, spk_id, book_no):
  df_transcript_path = os.path.join(root, '{0}-{1}.trans.txt'.format(spk_id, book_no))
  trans_columns = ['text']
  df_trans = pd.DataFrame([], columns=trans_columns)
  with open(df_transcript_path, 'r') as f:
    lines = f.readlines()
    for l in lines:
      l_splt = l.split(' ')
      fname = l_splt[0]
      text = ' '.join(l_splt[1:])
      df_trans = df_trans.append(pd.DataFrame(data=[[text]], index=[fname], columns=trans_columns))

  return(df_trans)

def create_metadata_librispeech():

  folder = '//vibe15/PublicAll/damcduff/SpeechSynthesis/LibriSpeech'
  folder_wav = os.path.join(folder,'train-clean-100')
  folder_data_ls = os.path.join(folder_data, 'librispeech')
  os.makedirs(folder_data_ls,exist_ok=True)

  meta_speakers = os.path.join(folder, 'SPEAKERS.TXT')
  df_speakers = pd.read_csv(meta_speakers,sep='|',skiprows=lambda x: x in list(range(11))+[44],index_col=0) #skip the description data and line 45 (has bad data)
  df_speakers.index.name = df_speakers.index.name[1:] #remove colon from index name

  columns = ['path', 'script', 'emt_label', 'spk_id', 'sex']
  df_metadata = pd.DataFrame([],columns=columns)
  for i,(root, dirs, files) in enumerate(os.walk(folder_wav, topdown=True)):
    if not(files): #check not empty
      continue
    spk_id = os.path.basename(os.path.dirname(root))
    book_no = os.path.basename(root)
    try:
      sex = df_speakers.loc[int(spk_id)].values[0][1]
    except KeyError:
      sex = 'N'

    #get transcripts in a dataframe
    df_trans = get_librispeech_transcript(root, spk_id, book_no)

    #get just the audio files
    audio_files = [f for f in files if f.endswith('.wav') or f.endswith('.flac')]

    for f in audio_files:
      path = os.path.join(spk_id, book_no,f)
      fname = os.path.basename(path)
      name = fname.split('.')[0]
      script = df_trans.loc[name].values[0]
      emt_label = 0
      df_metadata = df_metadata.append(pd.DataFrame([[path.replace('\\','/'),script[:-1],emt_label,spk_id,sex]],columns=columns),ignore_index=True)

    if i%100 ==0:
      print("Books Complete:",i)

  df_metadata_path = os.path.join(folder_data, 'metadata_librispeech.txt')
  df_metadata.to_csv(df_metadata_path,sep='|',header=False,index=False)

def create_metadata_vctk():

  folder = r'C:\Users\t-mawhit\Documents\data\VCTK-Corpus'
  folder_wav = os.path.join(folder,'wav48')
  folder_data_vctk = os.path.join(folder_data, 'vctk')
  os.makedirs(folder_data_vctk,exist_ok=True)

  #manually created this csv file because the original was really dumb!
  meta_speakers_path = os.path.join(folder, 'speaker-info.csv')
  df_speakers = pd.read_csv(meta_speakers_path,index_col=0)

  columns = ['path', 'script', 'emt_label', 'spk_id', 'sex','accent','region']
  df_metadata = pd.DataFrame([],columns=columns)
  emt_label=0
  cnt=0
  for i,(root, dirs, files) in enumerate(os.walk(folder_wav, topdown=True)):
    if not(files): #check not empty
      continue

    spk_name = os.path.basename(root)
    spk_id = int(spk_name[1:])

    try:
      row = df_speakers.loc[spk_id].values
      sex = row[1]
      accent = row[2]
      region = row[3]
    except:
      sex = 'N'
      accent = 'NA'
      region = 'NA'

    audio_files = [f for f in files if f.endswith('.wav') or f.endswith('.flac')]

    for f in audio_files:
      cnt+=1
      fname = f.split('.')[0]
      path = os.path.join('wav48', spk_name,f)
      script_path = os.path.join(folder,'txt', spk_name,fname+'.txt')
      try:
        with open(script_path) as f:
          script = f.read()
      except:
        print("couldn't find txt for", fname,"- skipping")
        continue

      if script.startswith('"'):
        script_before = script
        script = script[1:]
        print("clipping start quote - ",fname,"before:",script_before, "after:",script)
      if script.endswith('"'):
        script_before = script
        script = script[:-1]
        print("clipping end quote - ", fname, "before:", script_before, "after:", script)
      df_metadata = df_metadata.append(
        pd.DataFrame([[path.replace('\\', '/'), script[:-1], emt_label, spk_id, sex,accent,region]], columns=columns),
        ignore_index=True)
    print("finished", spk_name)

  df_metadata_path = os.path.join(folder_data, 'metadata_vctk.txt')
  df_metadata.to_csv(df_metadata_path,sep='|',header=False,index=False)

def vctk_metadata_accent():
  folder = r'C:\Users\t-mawhit\Documents\data\VCTK-Corpus'
  folder_wav = os.path.join(folder,'wav48')
  folder_data_vctk = os.path.join(folder_data, 'vctk')
  os.makedirs(folder_data_vctk,exist_ok=True)
  old_train_path = r'C:\Users\t-mawhit\Documents\code\Tacotron-2\data\vctk\train.txt'
  new_train_path = r'C:\Users\t-mawhit\Documents\code\Tacotron-2\data\vctk\train_accent.txt'

  #manually created this csv file because the original was really dumb!
  meta_speakers_path = os.path.join(folder, 'speaker-info.csv')
  df_speakers = pd.read_csv(meta_speakers_path,index_col=0)
  accents = df_speakers.ACCENTS.unique()

  accents = sorted(list(frozenset(accents)))

  new_meta = []
  with open(old_train_path, encoding='utf-8') as f:
    for i,line in enumerate(f):
      parts = line.strip().split('|')
      name=parts[10].split('_')[0][1:]
      try:
        parts[8] = accents.index(df_speakers.loc[int(name),'ACCENTS'])
      except KeyError:
        print("cound't find speaker:",name)
        continue
      new_meta.append(parts)

  with open(new_train_path, 'w', encoding='utf-8') as f:
    for m in new_meta:
      f.write('|'.join([str(x) for x in m]) + '\n')


if __name__ == '__main__':
  # create_metadata_vctk()
  # create_metadata_librispeech()
  # create_metadata_emt4()
  create_metadata_harriton()
  # create_metadata_jessa()
  # vctk_metadata_accent()

