import os
import pandas as pd

folder_data = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data')

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

if __name__ == '__main__':
  create_metadata_librispeech()
  # create_metadata_emt4()


