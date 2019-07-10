import os
import pandas as pd

folder_data = os.path.join(os.path.dirname(os.getcwd()), 'data')

def create_metadata_emt4():

  folder_wav = '//vibe15/PublicAll/STCM-101/Zo/Wav'
  folder_data_emt4 = os.path.join(folder_data, 'emt4')

  all_txt_path = os.path.join(folder_data_emt4, 'all_txt_wav.txt')
  df_all_txt = pd.read_csv(all_txt_path, sep='|', index_col=0,
                   names=['filename', 'script', 'emotion_label'])
  df_all_txt.emotion_label = df_all_txt.emotion_label.apply(int)

  paths = []
  for i,(root, dirs, files) in enumerate(os.walk(folder_wav, topdown=True)):
    for f in files:
      paths.append(os.path.join(root,f))

  columns = ['path', 'script', 'emt_label', 'spk_id']
  df_metadata = pd.DataFrame([],columns=columns)

  for p in paths[:]:
    fname = os.path.basename(p)
    name = int(fname.split('.')[0])
    path_no_ext = p.split('.')[0]
    row = df_all_txt.loc[name]
    script = row.script
    emt_label = row.emotion_label
    df_metadata = df_metadata.append(pd.DataFrame([[path_no_ext,script,emt_label,0]],columns=columns),ignore_index=True)
  df_metadata_path = os.path.join(folder_data, 'metadata_emt4.txt')
  df_metadata.to_csv(df_metadata_path,sep='|',header=False,index=False)

if __name__ == '__main__':
  create_metadata_emt4()


