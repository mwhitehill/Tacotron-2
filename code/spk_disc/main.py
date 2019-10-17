import tensorflow as tf
import os
from model import train, test, test_disc
from configuration import get_config
import argparse
import sys
sys.path.append(os.getcwd())

config = get_config()
tf.reset_default_graph()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=4, help='Number groups')
    parser.add_argument('--M', type=int, default=5, help='Number utterances per group')
    parser.add_argument('--gpu', type=str, default='0', help='gpu number to use')
    parser.add_argument('--remove_long_samps', action='store_true', default=False,
                        help='Will remove out the longest samples from EMT4/VCTK')
    parser.add_argument('--test_max_len', action='store_true', default=False,
                        help='Will create batches with the longest samples first to test max batch size')
    parser.add_argument('--TEST', action='store_true', default=False,
                        help='Uses small groups of batches to make testing faster')
    parser.add_argument('--train_filename', default='../data/train_emt4_jessa.txt')
    parser.add_argument('--model_type', default='emt', help='Options = emt or spk')
    parser.add_argument('--time_string', default=None, help='time string of previous saved model')
    parser.add_argument('--restore', action='store_true', default=False,
                        help='whether to restore the model')
    parser.add_argument('--discriminator', action='store_true', default=False,help='whether to use a discriminator as loss')
    args = parser.parse_args()
    if args.model_type == 'emt':
        print("setting N to 4 for training emotions")
        args.N = 4
    if args.model_type == 'spk' and 'emt4_jessa' in args.train_filename:
        print("setting N to 2 and M to 10 for training Zo/Jessa speaker")
        args.N = 2
        args.M = 10

    import socket
    if socket.gethostname() == 'area51.cs.washington.edu':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # start testing
    if args.TEST:
        print("\nTest session")
        if args.model_type == 'emt':
            MODEL_PATH = r'C:\Users\t-mawhit\Documents\code\Tacotron-2\tisv_model\checkpoints\zo_jessa_emt_disc'
        else:
            MODEL_PATH =r'C:\Users\t-mawhit\Documents\code\Tacotron-2\tisv_model\checkpoints\zo_jessa_spk_disc'

        # DATA_PATH = r'C:\Users\t-mawhit\Documents\code\Tacotron-2\eval\random\emt4_jessa_baseline_2\paired'#e40500_test_rs2_20samps'
        DATA_PATH = r'C:\Users\t-mawhit\Documents\code\Tacotron-2\eval\random\ej_ae_emb_disc_adv\2019.09.19_06-47-32'
        if os.path.isdir(MODEL_PATH):
            test_disc(MODEL_PATH,DATA_PATH, args)
        else:
            raise AssertionError("model path doesn't exist!")
    # start training
    else:
        print("\nTraining Session")
        os.makedirs(config.model_path, exist_ok=True)
        train(config.model_path, args)