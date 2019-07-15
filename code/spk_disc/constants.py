from pyaudio import paInt16
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Signal processing
SAMPLE_RATE = 16000
PREEMPHASIS_ALPHA = 0.97
FRAME_LEN = 0.025
FRAME_STEP = 0.01
NUM_FFT = 512
BUCKET_STEP = 1
MAX_SEC = 10

# Model
WEIGHTS_FILE = "spk_disc/weights.h5"
COST_METRIC = "cosine"  # euclidean or cosine
INPUT_SHAPE=(NUM_FFT,None,1)

# IO
ENROLL_LIST_FILE = "cfg/enroll_list.csv"
TEST_LIST_FILE = "cfg/test_list.csv"
RESULT_FILE = "res/results.csv"
