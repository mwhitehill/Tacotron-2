import tensorflow as tf
import os
from model import train, test
from configuration import get_config
import argparse

config = get_config()
tf.reset_default_graph()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--TEST', action='store_true', default=False, help='set if want to process test data, otherwise training data')
    args = parser.parse_args()

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
        train(config.model_path)


