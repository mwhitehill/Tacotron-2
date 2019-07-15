import tensorflow as tf
import os
import datetime

from networks import Emt_Disc
from hparams import hparams, hparams_debug_string
from tacotron.feeder import Feeder
import infolog
from infolog import log

def train(path, restore=False,restore_path = ''):

  checkpoint_path, log_path = path
  metadata_filename = 'C:/Users/t-mawhit/Documents/code/Tacotron-2/data/emt4/train.txt'

  setup_log(log_path, checkpoint_path, os.path.dirname(metadata_filename))

  coord = tf.train.Coordinator()
  feeder = Feeder(coord, metadata_filename, hparams)

  labels = tf.one_hot(feeder.emt_labels,4)
  labels_eval = tf.one_hot(feeder.eval_emt_labels,4)

  training = tf.placeholder(tf.bool)
  emt_disc = Emt_Disc(feeder.mel_targets, is_training=training, hparams=hparams)
  emt_disc_eval = Emt_Disc(feeder.eval_mel_targets, is_training=training, hparams=hparams)

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=emt_disc.logit, labels=labels))
  acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(emt_disc.logit, 1)), 'float32'))

  loss_val = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=emt_disc_eval.logit, labels=labels_eval))
  acc_val = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels_eval, 1), tf.argmax(emt_disc_eval.logit, 1)), 'float32'))

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    training_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

  saver = tf.train.Saver(max_to_keep=20)

  with tf.Session() as sess:
    if restore:
      saver.restore(sess, restore_path)

    sess.run(tf.global_variables_initializer())
    feeder.start_threads(sess)

    num_batch_ckpt = 10#64
    num_val_batches = feeder.test_steps
    num_batch_sav_ckpt = 20

    batches = 0
    total_loss = 0
    total_acc = 0
    total_loss_val = 0
    total_acc_val = 0

    while not coord.should_stop():
      cur_loss, cur_acc,  _ = sess.run([loss,acc,training_op],feed_dict={training:True})
      total_loss += cur_loss
      total_acc += cur_acc
      batches+=1

      if batches % num_batch_ckpt == 0:
        for i in range(num_val_batches):
          cur_loss_val, cur_acc_val = sess.run([loss_val,acc_val],feed_dict={training:False})
          total_loss_val += cur_loss_val
          total_acc_val += cur_acc_val
        message = "Batches Processed: {0} | Tr Loss: {1:5.3f} | Val Loss: {2:5.3f} | Tr Acc: {3:4.2f}% | Val Acc: {4:4.2f}%".format(batches,
                                                                                                                                    total_loss/num_batch_ckpt,
                                                                                                                                    total_loss_val / num_val_batches,
                                                                                                                                    total_acc * 100 / num_batch_ckpt,
                                                                                                                                    total_acc_val * 100 / num_val_batches)

        print(message)
        log(message, end='\r')
        total_loss = 0
        total_acc = 0
        total_loss_val = 0
        total_acc_val = 0

      if batches % num_batch_sav_ckpt == 0:
        saver.save(sess, checkpoint_path, global_step=batches)

def get_ckpt_save_path():
  path = os.getcwd()
  cur_time = datetime.datetime.now().strftime('%Y.%m.%d_%H-%M-%S')
  path = os.path.join(os.path.dirname(path),'checkpoints',cur_time)
  os.makedirs(path,exist_ok=True)
  ckpt_path = os.path.join(path, 'emt4_disc')
  log_path = os.path.join(path, 'emt4_disc.log')
  return(ckpt_path,log_path)

def setup_log(log_path, checkpoint_path, input_path):
  infolog.init(log_path, 'emt4_disc', None)
  log('hi')
  log('Checkpoint path: {}'.format(checkpoint_path))
  log('Loading training data from: {}'.format(input_path))
  log('Using model: {}'.format('emt4_disc'))
  log(hparams_debug_string())

if __name__ == '__main__':
  train(path=get_ckpt_save_path())