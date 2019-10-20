import tensorflow as tf
import numpy as np
import os
import time
from utils import Feeder, normalize, similarity, loss_cal, optim, test_batch
from configuration import get_config
import sys
sys.path.append(os.getcwd())
from tacotron.models.modules import ReferenceEncoder
from tacotron.utils import ValueWindow
from tensorflow.contrib import rnn
import datetime
from hparams import hparams
import pandas as pd
import scikitplot as skplt
import matplotlib.pyplot as plt

VAL_ITERS = 5

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

def train(path, args):
    tf.reset_default_graph()    # reset graph
    timestamp = time_string() if args.time_string == None else args.time_string

    # draw graph
    feeder = Feeder(args.train_filename, args, hparams)

    output_classes = max([int(f) for f in feeder.total_emt])+1 if args.model_type in ['emt', 'accent'] else max([int(f) for f in feeder.total_spk])+1

    batch = tf.placeholder(shape= [args.N*args.M, None, config.n_mels], dtype=tf.float32)  # input batch (time x batch x n_mel)
    labels = tf.placeholder(shape=[args.N * args.M],dtype=tf.int32)
    lr = tf.placeholder(dtype= tf.float32)  # learning rate
    global_step = tf.Variable(0, name='global_step', trainable=False)
    w = tf.get_variable("w", initializer= np.array([10], dtype=np.float32))
    b = tf.get_variable("b", initializer= np.array([-5], dtype=np.float32))

    # embedded = triple_lstm(batch)
    print("Training {} Discriminator Model".format(args.model_type))
    encoder = ReferenceEncoder(filters=hparams.reference_filters, kernel_size=(3, 3),
                               strides=(2, 2),is_training=True,
                               scope='Tacotron_model/inference/pretrained_ref_enc_{}'.format(args.model_type), depth=hparams.reference_depth)  # [N, 128])
    embedded = encoder(batch)
    embedded = normalize(embedded)

    if args.discriminator:
        logit = tf.layers.dense(embedded, output_classes, name='Tacotron_model/inference/pretrained_ref_enc_{}_dense'.format(args.model_type))
        labels_one_hot = tf.one_hot(tf.to_int32(labels), output_classes)
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit,labels=labels_one_hot))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit,labels=labels_one_hot))
        acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(labels_one_hot, 1),predictions=tf.argmax(logit, 1))
        val_acc, val_acc_op = tf.metrics.accuracy(labels=tf.argmax(labels_one_hot, 1), predictions=tf.argmax(logit, 1))
    else:
        # loss
        sim_matrix = similarity(embedded, w, b, args.N, args.M, P=hparams.reference_depth)
        print("similarity matrix size: ", sim_matrix.shape)
        loss = loss_cal(sim_matrix, args.N, args.M, type=config.loss)
        val_acc_op = tf.constant(1.)

    # optimizer operation
    trainable_vars= tf.trainable_variables()                # get variable list
    optimizer= optim(lr)                                    # get optimizer (type is determined by configuration)
    grads, vars= zip(*optimizer.compute_gradients(loss))    # compute gradients of variables with respect to loss

    if args.discriminator:
        grads_rescale = grads
    else:
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
    loss_window = ValueWindow(100)
    acc_window = ValueWindow(100)
    val_loss_window = ValueWindow(5)
    val_acc_window = ValueWindow(5)

    # training session
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        checkpoint_folder = os.path.join(path, "checkpoints",timestamp)
        logs_folder = os.path.join(path, "logs", timestamp)
        os.makedirs(checkpoint_folder, exist_ok=True)  # make folder to save model
        os.makedirs(logs_folder, exist_ok=True)        # make folder to save log
        model_name = '{}_disc_model.ckpt'.format(args.model_type)
        checkpoint_path = os.path.join(checkpoint_folder, model_name)

        if args.restore:
            checkpoint_state = tf.train.get_checkpoint_state(checkpoint_folder)
            if (checkpoint_state and checkpoint_state.model_checkpoint_path):
                print('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path))
                saver.restore(sess, checkpoint_state.model_checkpoint_path)
            else:
                print('No model to load at {}'.format(checkpoint_folder))
                saver.save(sess, checkpoint_path, global_step=global_step)
        else:
            print('Starting new training!')
            saver.save(sess, checkpoint_path, global_step=global_step)


        writer = tf.summary.FileWriter(logs_folder, sess.graph)
        lr_factor = 1   # lr decay factor ( 1/2 per 10000 iteration)

        iterations = 30000 if args.model_type == 'emt' else config.iteration
        for iter in range(iterations):
            if args.discriminator:
                batch_iter, _, labels_iter = feeder.random_batch_disc()
            else:
                batch_iter, _, labels_iter = feeder.random_batch()
            # run forward and backward propagation and update parameters
            step, _, loss_cur, summary, acc_cur = sess.run([global_step, train_op, loss, merged, acc_op],
                                  feed_dict={batch:batch_iter, labels:labels_iter, lr: config.lr*lr_factor})

            loss_window.append(loss_cur)
            acc_window.append(acc_cur)

            if step % 10 == 0:
                writer.add_summary(summary, step)   # write at tensorboard
            if (step+1) % 20 == 0:
                val_loss_cur_batch = 0
                val_acc_cur_batch = 0
                for iter in range(VAL_ITERS):
                    if args.discriminator:
                        batch_iter, _, labels_iter = feeder.random_batch_disc(TEST=True)
                    else:
                        batch_iter, _, labels_iter = feeder.random_batch(TEST=True)
                    # run forward and backward propagation and update parameters
                    val_loss_cur, val_acc_cur = sess.run([loss, val_acc_op], feed_dict={batch: batch_iter, labels: labels_iter})
                    val_loss_cur_batch += val_loss_cur
                    val_acc_cur_batch += val_acc_cur
                val_loss_cur_batch /= VAL_ITERS
                val_acc_cur_batch /= VAL_ITERS
                val_loss_window.append(val_loss_cur_batch)
                val_acc_window.append(val_acc_cur_batch)

                message = "(iter : %d) loss: %.4f" % ((step+1),loss_window.average)
                if args.discriminator:
                    message += ', acc: {:.2f}%'.format(acc_window.average)
                message += ", val_loss: %.4f" % (val_loss_window.average)
                if args.discriminator:
                    message += ', val_acc: {:.2f}%'.format(val_acc_window.average)
                print(message)

            lr_changed=False
            if args.model_type == 'emt':
                if step > 6000:
                    lr_changed = True if lr_factor != .01 else False
                    lr_factor = .01
                elif step > 4000:
                    lr_changed = True if lr_factor != .1 else False
                    lr_factor = .1
                if lr_changed:
                    print("learning rate is decayed! current lr : ", config.lr * lr_factor)
            elif args.model_type == 'spk':
                if step > 300:#4000:
                    lr_changed = True if lr_factor != .01 else False
                    lr_factor = .01
                elif step > 180:#2500:
                    lr_changed = True if lr_factor != .1 else False
                    lr_factor = .1
                if lr_changed:
                    print("learning rate is decayed! current lr : ", config.lr * lr_factor)
            if step % config.save_checkpoint_iters == 0:
                saver.save(sess, checkpoint_path, global_step=global_step)

def test_disc(path_model, path_meta, path_data, args):

    #dataset|audio_filename|mel_filename|linear_filename|spk_emb_filename|time_steps|mel_frames|text|emt_label|spk_label|basename|emt_name|emt_file|spk_name|spk_file

    df = pd.read_csv(path_meta, sep='|')
    n_samps = len(df.index)

    tf.reset_default_graph()  # reset graph

    # draw graph
    feeder = Feeder(args.train_filename, args, hparams)

    output_classes = max([int(f) for f in feeder.total_emt]) + 1 if args.model_type in ['emt', 'accent'] else max(
        [int(f) for f in feeder.total_spk]) + 1

    batch = tf.placeholder(shape=[n_samps, None, config.n_mels],
                           dtype=tf.float32)  # input batch (time x batch x n_mel)
    labels = tf.placeholder(shape=[n_samps], dtype=tf.int32)

    # embedded = triple_lstm(batch)
    print("Testing {} Discriminator Model".format(args.model_type))
    encoder = ReferenceEncoder(filters=hparams.reference_filters, kernel_size=(3, 3),
                               strides=(2, 2), is_training=True, scope='Tacotron_model/inference/pretrained_ref_enc_{}'.format(args.model_type),
                               depth=hparams.reference_depth)  # [N, 128])
    embedded = encoder(batch)
    embedded = normalize(embedded)

    logit = tf.layers.dense(embedded, output_classes,name='Tacotron_model/inference/pretrained_ref_enc_{}_dense'.format(args.model_type))
    logit_sm = tf.nn.softmax(logit)
    labels_one_hot = tf.one_hot(tf.to_int32(labels), output_classes)
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit,labels=labels_one_hot))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=labels_one_hot))
    acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(labels_one_hot, 1), predictions=tf.argmax(logit, 1))
    batch_size = tf.shape(labels)[0]
    #acc_e, acc_op_e = tf.metrics.accuracy(labels=tf.argmax(labels_one_hot[:batch_size], 1), predictions=tf.argmax(logit[:batch_size], 1))


    saver = tf.train.Saver()

    # training session
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        checkpoint_state = tf.train.get_checkpoint_state(path_model)
        print('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path))
        saver.restore(sess, checkpoint_state.model_checkpoint_path)



        batch_iter, _, labels_iter = test_batch(path_data, df, args)

        # run forward and backward propagation and update parameters
        loss_cur, acc_cur, lbls, log, emb = sess.run(
            [loss, acc_op, labels, logit_sm, embedded],
            feed_dict={batch: batch_iter, labels: labels_iter})
        print('loss: {:.4f}, acc: {:.2f}%'.format(loss_cur, acc_cur))
        #print(np.max(log, 1))
        #print(np.mean(np.max(log, 1)))
        preds = np.argmax(log,1)
        print(preds)
        print(lbls)
        df_results = pd.DataFrame([])
        df_results['preds'] = preds
        df_results['true'] = lbls
        batch_size = len(df_results.index)
        df_results['dataset'] = 'emt4'
        df_results.loc[df_results.index >= batch_size//2,'dataset']='jessa'
        df_results.to_csv(os.path.join(path_data,'results.csv'))

        df_jessa = df_results[df_results.dataset=='jessa']
        pred_cnf = df_jessa.preds
        true_cnf = df_jessa.true
        emotions = ['neutral','angry','happy','sad']
        for i,emt in enumerate(emotions):
            pred_cnf = np.where(pred_cnf == i, emt, pred_cnf)
            true_cnf = np.where(true_cnf == i, emt, true_cnf)

        skplt.metrics.plot_confusion_matrix(true_cnf, pred_cnf, labels=emotions, normalize=True)
        plt.show()
        
        emb_path = os.path.join(path_data, 'emb.csv')
        emb_meta_path = os.path.join(path_data, 'emb_meta.csv')
        pd.DataFrame(emb).to_csv(emb_path,sep='\t',index=False,header=False)
        columns_to_keep = ['dataset', 'mel_filename', 'mel_frames', 'emt_label', 'spk_label', 'basename', 'sex']
        df = df.loc[:, columns_to_keep]
        paired_label = 'paired' if 'paired' in path_data else 'unpaired'
        df['paired'] = paired_label
        df.to_csv(emb_meta_path,sep='\t',index=False)

def get_embeddings(path, args):
    tf.reset_default_graph()    # reset graph
    if args.time_string == None:
        raise ValueError('must provide valid time_string')

    emb_dir = os.path.join(path, 'embeddings')
    os.makedirs(emb_dir, exist_ok=True)
    meta_path = os.path.join(emb_dir, 'meta.tsv')

    emb_path = os.path.join(emb_dir, 'emb_emt.tsv') if args.model_type == 'emt' else os.path.join(emb_dir, 'emb_spk.tsv')


    # draw graph
    feeder = Feeder(args.train_filename, args, hparams)
    datasets = ['emt4','vctk'] if args.model_type =='emt' else ['vctk']
    num_datasets = len(datasets)

    batch = tf.placeholder(shape= [num_datasets * args.N*args.M, None, config.n_mels], dtype=tf.float32)  # input batch (time x batch x n_mel)
    w = tf.get_variable("w", initializer= np.array([10], dtype=np.float32))
    b = tf.get_variable("b", initializer= np.array([-5], dtype=np.float32))

    # embedded = triple_lstm(batch)
    print("{} Discriminator Model".format(args.model_type))
    encoder = ReferenceEncoder(filters=hparams.reference_filters, kernel_size=(3, 3),
                               strides=(2, 2), is_training=True, scope='Tacotron_model/inference/pretrained_ref_enc_{}'.format(args.model_type),
                               depth=hparams.reference_depth)  # [N, 128])
    embedded = encoder(batch)

    # loss
    sim_matrix = similarity(embedded, w, b, num_datasets*args.N, args.M, P=hparams.reference_depth)
    print("similarity matrix size: ", sim_matrix.shape)
    loss = loss_cal(sim_matrix, num_datasets * args.N, args.M, type=config.loss)

    saver = tf.train.Saver()

    # training session
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        checkpoint_folder = os.path.join(path, "checkpoints",args.time_string)

        checkpoint_state = tf.train.get_checkpoint_state(checkpoint_folder)
        if (checkpoint_state and checkpoint_state.model_checkpoint_path):
            print('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path))
            saver.restore(sess, checkpoint_state.model_checkpoint_path)
        else:
            raise ValueError('No model to load at {}'.format(checkpoint_folder))
        feeder_batch, meta = feeder.emb_batch(make_meta=True, datasets=datasets)
        emb, loss = sess.run([embedded, loss], feed_dict={batch: feeder_batch})
        print("loss: {:.4f}".format(loss))
        meta.to_csv(meta_path, sep='\t', index=False)
        pd.DataFrame(emb).to_csv(emb_path, sep='\t', index=False, header=False)

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

    import argparse
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
    args = parser.parse_args()

    args.M=10
    get_embeddings(config.model_path, args)

    # folder = r'C:\Users\t-mawhit\Documents\code\Speaker_Verification\tisv_model\checkpoints\2019.07.18_17-13-28'
    # MODEL_PATH = tf.train.get_checkpoint_state(checkpoint_dir=folder).all_model_checkpoint_paths[-1]
    # get_embedding_test(MODEL_PATH)

    # mel_spec_test = np.zeros()
    # get_embedding_preprocess(mel_spec)

    # None, config.N * config.M, config.n_mels]
    pass
