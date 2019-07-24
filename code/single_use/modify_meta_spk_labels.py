import os

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
			parts[8] = spk_ids.index(parts[8])+1 #reserving emt4 as spk label 0
			f_out.write('|'.join([str(x) for x in parts]) + '\n')

print("wrote",i,"new lines")
