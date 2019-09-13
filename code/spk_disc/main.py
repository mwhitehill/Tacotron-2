import tensorflow as tf
import os
from model import train, test
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
    parser.add_argument('--remove_long_samps', action='store_true', default=False,
                        help='Will remove out the longest samples from EMT4/VCTK')
    parser.add_argument('--test_max_len', action='store_true', default=False,
                        help='Will create batches with the longest samples first to test max batch size')
    parser.add_argument('--TEST', action='store_true', default=False,
                        help='Uses small groups of batches to make testing faster')
    parser.add_argument('--train_filename', default='../data/train_emt4_vctk_e40_v15.txt')
    parser.add_argument('--model_type', default='emt', help='Options = emt or spk')
    parser.add_argument('--time_string', default=None, help='time string of previous saved model')
    parser.add_argument('--restore', action='store_true', default=False,
                        help='whether to restore the model')
    parser.add_argument('--discriminator', action='store_true', default=False,help='whether to use a discriminator as loss')
    parser.add_argument('--output_classes', type=int, default=5, help='# classes for discriminator')
    args = parser.parse_args()
    if args.model_type == 'emt':
        print("setting N to 4 for training emotions")
        args.N = 4

    # start testing
    if args.TEST:
        print("\nTest session")
        if os.path.isdir(config.model_path,exist_ok=True):
            test(config.model_path)
        else:
            raise AssertionError("model path doesn't exist!")
    # start training
    else:
        print("\nTraining Session")
        os.makedirs(config.model_path, exist_ok=True)
        train(config.model_path, args)