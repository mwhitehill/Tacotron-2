import tensorflow as tf
import numpy as np
import os
import time
from utils import random_batch, normalize, similarity, loss_cal, optim
from configuration import get_config
from tensorflow.contrib import rnn
import datetime

config = get_config()

def time_string():
  return datetime.datetime.now().strftime('%Y.%m.%d_%H-%M-%S')


def triple_lstm(batch):

    # embedding lstm (3-layer default)
    with tf.variable_scope("spk_emb_lstm"):
        lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=config.hidden, num_proj=config.proj) for i in range(config.num_layer)]
        lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)    # define lstm op and variables
        outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=batch, dtype=tf.float32, time_major=True)   # for TI-VS must use dynamic rnn
        embedded = outputs[-1]                            # the last ouput is the embedded d-vector
        embedded = normalize(embedded)                    # normalize
    print("embedded size: ", embedded.shape)

    return(embedded)

def train(path):
    tf.reset_default_graph()    # reset graph
    timestamp = time_string()

    # draw graph
    batch = tf.placeholder(shape= [None, config.N*config.M, config.n_mels], dtype=tf.float32)  # input batch (time x batch x n_mel)
    lr = tf.placeholder(dtype= tf.float32)  # learning rate
    global_step = tf.Variable(0, name='global_step', trainable=False)
    w = tf.get_variable("w", initializer= np.array([10], dtype=np.float32))
    b = tf.get_variable("b", initializer= np.array([-5], dtype=np.float32))

    embedded = triple_lstm(batch)

    # loss
    sim_matrix = similarity(embedded, w, b)
    print("similarity matrix size: ", sim_matrix.shape)
    loss = loss_cal(sim_matrix, type=config.loss)

    # optimizer operation
    trainable_vars= tf.trainable_variables()                # get variable list
    optimizer= optim(lr)                                    # get optimizer (type is determined by configuration)
    grads, vars= zip(*optimizer.compute_gradients(loss))    # compute gradients of variables with respect to loss
    grads_clip, _ = tf.clip_by_global_norm(grads, 3.0)      # l2 norm clipping by 3
    grads_rescale= [0.01*grad for grad in grads_clip[:2]] + grads_clip[2:]   # smaller gradient scale for w, b
    train_op= optimizer.apply_gradients(zip(grads_rescale, vars), global_step= global_step)   # gradient update operation

    # check variables memory
    variable_count = np.sum(np.array([np.prod(np.array(v.get_shape().as_list())) for v in trainable_vars]))
    print("total variables :", variable_count)

    # record loss
    loss_summary = tf.summary.scalar("loss", loss)
    merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=20)

    # training session
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        checkpoint_folder = os.path.join(path, "checkpoints",timestamp)
        logs_folder = os.path.join(path, "logs", timestamp)
        os.makedirs(checkpoint_folder, exist_ok=True)  # make folder to save model
        os.makedirs(logs_folder, exist_ok=True)        # make folder to save log

        writer = tf.summary.FileWriter(logs_folder, sess.graph)
        lr_factor = 1   # lr decay factor ( 1/2 per 10000 iteration)
        loss_acc = 0    # accumulated loss ( for running average of loss)

        for iter in range(config.iteration):
            # run forward and backward propagation and update parameters
            _, loss_cur, summary = sess.run([train_op, loss, merged],
                                  feed_dict={batch: random_batch(), lr: config.lr*lr_factor})
            print("iter:", iter, "loss:", loss_cur)

            loss_acc += loss_cur    # accumulated loss for each 100 iteration

            if iter % 10 == 0:
                writer.add_summary(summary, iter)   # write at tensorboard
            if (iter+1) % 20 == 0:
                print("(iter : %d) loss: %.4f" % ((iter+1),loss_acc/20))
                loss_acc = 0                        # reset accumulated loss
            if (iter+1) % config.decay_lr_iters == 0:
                lr_factor /= 2                      # lr decay
                print("learning rate is decayed! current lr : ", config.lr*lr_factor)
            if (iter+1) % config.save_checkpoint_iters == 0:
                saver.save(sess, os.path.join(checkpoint_folder, "model.ckpt"), global_step=iter)
                print("model is saved!")

def get_embedding(mel_spec):



# Get Embedding for an input
def embedding_model(batch):
    embedded = triple_lstm(batch)
    return(embedded)

def get_embedding_test(MODEL_PATH):

    tf.reset_default_graph()
    batch = tf.placeholder(shape=[None, config.N * config.M, config.n_mels], dtype=tf.float32)
    get_embedding = embedding_model(batch)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        vars_to_restore = [v for v in tf.global_variables() if v.name.split('/')[0] == 'spk_emb_lstm']

        saver_spk_emb = tf.train.Saver(var_list=vars_to_restore)
        saver_spk_emb.restore(sess, MODEL_PATH)

        test_batch = np.zeros([140,config.N*config.M, config.n_mels])
        embedded = sess.run(get_embedding,feed_dict={batch:test_batch})
        print(embedded.shape)

# def get_embeddings_from_csv():
#
# 	list_file = r'data/embedding/file_list.csv'
# 	emb_file = r'data/embedding/embedding.tsv'
# 	meta_file = r'data/embedding/metadata.tsv'
#
# 	print("Loading model weights from [{}]....".format(WEIGHTS_FILE))
# 	model = vggvox_model()
# 	model.load_weights(WEIGHTS_FILE)
# 	# model.summary()
#
# 	print("Processing samples....")
# 	buckets = build_buckets(MAX_SEC, BUCKET_STEP, FRAME_STEP)
# 	meta = pd.read_csv(list_file, delimiter=",")
# 	result = meta.copy()
# 	result['features'] = result['filename'].apply(lambda x: get_fft_spectrum(x, buckets))
# 	result['embedding'] = result['features'].apply(lambda x: np.squeeze(model.predict(x.reshape(1,*x.shape,1))))
#
# 	#expand list to columns
# 	embs = result['embedding'].apply(pd.Series)
#
# 	#get just filename, not full path
# 	meta['filename'] = meta['filename'].apply(lambda x: repr(x).split(r'\\')[-1][:-1])
#
# 	#print to csv for use with the tensorflow embedding projector - https://projector.tensorflow.org/
# 	embs.to_csv(emb_file,sep='\t',index=False,header=False)
# 	meta.to_csv(meta_file, sep='\t',index=False)


# Test Session
def test(path):
    tf.reset_default_graph()

    # draw graph
    enroll = tf.placeholder(shape=[None, config.N*config.M, config.n_mels], dtype=tf.float32) # enrollment batch (time x batch x n_mel)
    verif = tf.placeholder(shape=[None, config.N*config.M, config.n_mels], dtype=tf.float32)  # verification batch (time x batch x n_mel)
    batch = tf.concat([enroll, verif], axis=1)

    embedded = triple_lstm(batch)

    # enrollment embedded vectors (speaker model)
    enroll_embed = normalize(tf.reduce_mean(tf.reshape(embedded[:config.N*config.M, :], shape= [config.N, config.M, -1]), axis=1))
    # verification embedded vectors
    verif_embed = embedded[config.N*config.M:, :]

    similarity_matrix = similarity(embedded=verif_embed, w=1., b=0., center=enroll_embed)

    saver = tf.train.Saver(var_list=tf.global_variables())
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # load model
        print("model path :", path)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=os.path.join(path, "Check_Point"))
        ckpt_list = ckpt.all_model_checkpoint_paths
        loaded = 0
        for model in ckpt_list:
            if config.model_num == int(model[-1]):    # find ckpt file which matches configuration model number
                print("ckpt file is loaded !", model)
                loaded = 1
                saver.restore(sess, model)  # restore variables from selected ckpt file
                break

        if loaded == 0:
            raise AssertionError("ckpt file does not exist! Check config.model_num or config.model_path.")

        print("test file path : ", config.test_path)

        # return similarity matrix after enrollment and verification
        time1 = time.time() # for check inference time
        if config.tdsv:
            S = sess.run(similarity_matrix, feed_dict={enroll:random_batch(shuffle=False, noise_filenum=1, TEST=True),
                                                       verif:random_batch(shuffle=False, noise_filenum=2,TEST=True)})
        else:
            S = sess.run(similarity_matrix, feed_dict={enroll:random_batch(shuffle=False,TEST=True),
                                                       verif:random_batch(shuffle=False, utter_start=config.M, TEST=True)})
        S = S.reshape([config.N, config.M, -1])
        time2 = time.time()

        np.set_printoptions(precision=2)
        print("inference time for %d utterences : %0.2fs"%(2*config.M*config.N, time2-time1))
        print(S)    # print similarity matrix

        # calculating EER
        diff = 1; EER=0; EER_thres = 0; EER_FAR=0; EER_FRR=0

        # through thresholds calculate false acceptance ratio (FAR) and false reject ratio (FRR)
        for thres in [0.01*i+0.5 for i in range(50)]:
            S_thres = S>thres

            # False acceptance ratio = false acceptance / mismatched population (enroll speaker != verification speaker)
            FAR = sum([np.sum(S_thres[i])-np.sum(S_thres[i,:,i]) for i in range(config.N)])/(config.N-1)/config.M/config.N

            # False reject ratio = false reject / matched population (enroll speaker = verification speaker)
            FRR = sum([config.M-np.sum(S_thres[i][:,i]) for i in range(config.N)])/config.M/config.N

            # Save threshold when FAR = FRR (=EER)
            if diff> abs(FAR-FRR):
                diff = abs(FAR-FRR)
                EER = (FAR+FRR)/2
                EER_thres = thres
                EER_FAR = FAR
                EER_FRR = FRR

        print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)"%(EER,EER_thres,EER_FAR,EER_FRR))


if __name__ == "__main__":

    # folder = r'C:\Users\t-mawhit\Documents\code\Speaker_Verification\tisv_model\checkpoints\2019.07.18_17-13-28'
    # MODEL_PATH = tf.train.get_checkpoint_state(checkpoint_dir=folder).all_model_checkpoint_paths[-1]
    # get_embedding_test(MODEL_PATH)

    mel_spec_test = np.zeros()
    get_embedding_preprocess(mel_spec)

    # None, config.N * config.M, config.n_mels]


