import numpy as np
import os
import pandas as pd
from shutil import copyfile
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def human_test_randomize():
    folder = r'C:\Users\Administrator\Documents\code\Tacotron-2\eval\human_test'
    folder_new = os.path.join(folder, 'new')
    folder_bsl = os.path.join(folder, 'bsl')
    folder_test = os.path.join(folder, 'test')
    save_path = os.path.join(folder,'human_test_answers.csv')
    files = [f for f in os.listdir(folder_new) if f.endswith('wav')]

    np.random.shuffle(files)
    columns = ['files','test1','test2']
    df = pd.DataFrame([],columns=columns)
    for i,f in enumerate(files):

        choice = np.random.choice(range(2),1)[0]

        for j in range(2):
            dst = os.path.join(folder_test,'test_{}_{}.wav'.format(i,j))
            if j == 0:
                src = os.path.join(folder_new, f) if choice else os.path.join(folder_bsl,f)
            else:
                src = os.path.join(folder_bsl, f) if choice else os.path.join(folder_new, f)
            copyfile(src, dst)

        test1 = 'new' if choice else 'bsl'
        test2 = 'bsl' if choice else 'new'

        df = df.append(pd.DataFrame([[f,test1,test2]],columns=columns),ignore_index=True)

    df.to_csv(save_path)
    # pd.DataFrame(files,columns=['files']).to_csv(os.path.join(folder,'files.csv'))
    # np.savetxt(os.path.join(folder,'files.txt'),files,fmt='%s')

def cnf_matrix(DSET='jessa', NEW=True, EMT=True):

    assert(DSET in ['jessa','both'])

    folder = r'C:\Users\Administrator\Documents\code\Tacotron-2\results'

    new_suff = 'new' if NEW else 'bsl'
    model_type_suff = 'emt' if EMT else 'spk'

    folder_full = os.path.join(folder,'{}_{}'.format(model_type_suff,DSET),new_suff)
    path = os.path.join(folder_full, 'results_{}_{}.csv'.format(new_suff, model_type_suff))
    if not(os.path.exists(path)):
        path = os.path.join(folder_full, 'results.csv')
    df = pd.read_csv(path)
    preds = df.preds.astype(str)
    true = df.true.astype(str)
    labels = ['neutral', 'angry', 'happy', 'sad'] if EMT else ['emt4', 'jessa']
    for i, emt in enumerate(labels):
        preds = np.where(preds == str(i), emt, preds)
        true = np.where(true == str(i), emt, true)

    skplt.metrics.plot_confusion_matrix(true, preds, labels=labels, normalize=False)
    # plt.savefig('cm.svg', format='svg')
    plt.show()

def plot_2d_clusters(features, labels, figsize=(12, 12), name='cluster', save=False, show=True, legend=True,
                     linewidth='1'):
    label_set = set(labels)
    plt.figure()#figsize=figsize)
    colors = ['r','b','m','g','k'] if len(label_set) <=6 else plt.cm.rainbow(np.linspace(0, 1, len(label_set)))

    for c_pid, (c_color, c_label) in enumerate(zip(colors, list(label_set))):
        label_idxs = np.where(labels == c_label)
        plt.scatter(features[label_idxs, 0],
                    features[label_idxs, 1],
                    marker='.', color=c_color,
                    linewidth=linewidth, alpha=0.8,
                    label=c_label)
    plt.xlim([-8,8])
    plt.ylim([-6,6])
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    # plt.title(name)
    if legend:
        plt.legend(loc='best')
    # if save:
        # save_title = 'spk_only' if SPEECH_ONLY else 'spk_and_coughs'
        # plt.savefig('./embeddings/pics/{}_{}_tsne.jpg'.format(save_title,CV))
    plt.show()

def plot_embs(DSET='jessa', NEW=True, EMT=True):

    assert (DSET in ['jessa', 'both'])

    folder = r'C:\Users\Administrator\Documents\code\Tacotron-2\results'

    new_suff = 'new' if NEW else 'bsl'
    model_type_suff = 'emt' if EMT else 'spk'

    folder_full = os.path.join(folder, '{}_{}'.format(model_type_suff, DSET), new_suff)
    path_emb = os.path.join(folder_full, 'emb_100_{}_{}.csv'.format(new_suff, model_type_suff))
    path_meta = os.path.join(folder_full, 'emb_meta_100_{}_{}.csv'.format(new_suff, model_type_suff))
    if not (os.path.exists(path_emb)):
        path_emb = os.path.join(folder_full, 'emb_100.csv')
        path_meta = os.path.join(folder_full, 'emb_meta_100.csv')
    emb = pd.read_csv(path_emb,sep=r'\t',header=None)
    meta = pd.read_csv(path_meta,sep=r'\t')

    labels = meta.emt_label if EMT else meta.dataset
    if EMT:
        labels = labels.astype(str)
        emotions = ['neutral', 'angry', 'happy', 'sad']
        for i, emt in enumerate(emotions):
            labels = np.where(labels == str(i), emt, labels)


    tsne_obj = TSNE(n_components=2, init='pca',method='barnes_hut', n_iter=10000, verbose=1)
    tsne_features = tsne_obj.fit_transform(emb)

    linewidth = '4'
    plot_2d_clusters(tsne_features,labels, name='t-SNE Embedding Clustering', save=False, legend=True,
                     linewidth=linewidth)


if __name__ == "__main__":
    # human_test_randomize()
    DSET='jessa' #'both'
    NEW=False
    EMT=True
    # cnf_matrix(DSET=DSET,NEW=NEW,EMT=EMT)
    plot_embs(DSET=DSET,NEW=NEW,EMT=EMT)