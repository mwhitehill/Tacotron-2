import argparse
import os
from time import sleep

import infolog
import tensorflow as tf
from hparams import hparams
from infolog import log
from tacotron.synthesize import tacotron_synthesize
from tacotron.train import tacotron_train
# from wavenet_vocoder.train import wavenet_train

log = infolog.log


def save_seq(file, sequence, input_path):
	'''Save Tacotron-2 training state to disk. (To skip for future runs)
	'''
	sequence = [str(int(s)) for s in sequence] + [input_path]
	with open(file, 'w') as f:
		f.write('|'.join(sequence))

def read_seq(file):
	'''Load Tacotron-2 training state from disk. (To skip if not first run)
	'''
	if os.path.isfile(file):
		with open(file, 'r') as f:
			sequence = f.read().split('|')

		return [bool(int(s)) for s in sequence[:-1]], sequence[-1]
	else:
		return [0, 0, 0], ''

def prepare_run(args):
	modified_hp = hparams.parse(args.hparams)
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
	run_name = args.name or args.model
	log_dir = '../logs-{}'.format(run_name)
	os.makedirs(log_dir, exist_ok=True)
	infolog.init(os.path.join(log_dir, 'Terminal_train_log'), run_name, args.slack_url)
	return log_dir, modified_hp

def train(args, log_dir, hparams):
	state_file = os.path.join(log_dir, 'state_log')
	#Get training states
	(taco_state, GTA_state, wave_state), input_path = read_seq(state_file)
	if not taco_state:
		log('\n#############################################################\n')
		log('Tacotron Train\n')
		log('###########################################################\n')
		checkpoint = tacotron_train(args, log_dir, hparams)
		tf.reset_default_graph()
		#Sleep 1/2 second to let previous graph close and avoid error messages while synthesis
		sleep(0.5)
		if checkpoint is None:
			raise('Error occured while training Tacotron, Exiting!')
		taco_state = 1
		save_seq(state_file, [taco_state, GTA_state, wave_state], input_path)
	else:
		checkpoint = os.path.join(log_dir, 'taco_pretrained/')

	if not GTA_state:
		log('\n#############################################################\n')
		log('Tacotron GTA Synthesis\n')
		log('###########################################################\n')
		input_path = tacotron_synthesize(args, hparams, checkpoint)
		tf.reset_default_graph()
		#Sleep 1/2 second to let previous graph close and avoid error messages while Wavenet is training
		sleep(0.5)
		GTA_state = 1
		save_seq(state_file, [taco_state, GTA_state, wave_state], input_path)
	else:
		input_path = os.path.join('tacotron_' + args.output_dir, 'gta', 'map.txt')

	if input_path == '' or input_path is None:
		raise RuntimeError('input_path has an unpleasant value -> {}'.format(input_path))

	if not wave_state:
		log('\n#############################################################\n')
		log('Wavenet Train\n')
		log('###########################################################\n')
		raise("not using wavent right now")
		# checkpoint = wavenet_train(args, log_dir, hparams, input_path)
		# if checkpoint is None:
		# 	raise ('Error occured while training Wavenet, Exiting!')
		# wave_state = 1
		# save_seq(state_file, [taco_state, GTA_state, wave_state], input_path)

	if wave_state and GTA_state and taco_state:
		log('TRAINING IS ALREADY COMPLETE!!')

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', default='')
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--tacotron_input', default='../data/train_emt4_jessa.txt')
	parser.add_argument('--wavenet_input', default='tacotron_output/gta/map.txt')
	parser.add_argument('--name', help='Name of logging directory.')
	parser.add_argument('--model', default='Tacotron-2')
	parser.add_argument('--input_dir', default='../data', help='folder to contain inputs sentences/targets')
	parser.add_argument('--output_dir', default='output', help='folder to contain synthesized mel spectrograms')
	parser.add_argument('--mode', default='synthesis', help='mode for synthesis of tacotron after training')
	parser.add_argument('--GTA', default='True', help='Ground truth aligned synthesis, defaults to True, only considered in Tacotron synthesis mode')
	parser.add_argument('--restore', action='store_true', default=False, help='Set this to False to do a fresh training')
	parser.add_argument('--summary_interval', type=int, default=1000000,
		help='Steps between running summary ops')
	parser.add_argument('--embedding_interval', type=int, default=1000000,
		help='Steps between updating embeddings projection visualization')
	parser.add_argument('--checkpoint_interval', type=int, default=250,
		help='Steps between writing checkpoints')
	parser.add_argument('--eval_interval', type=int, default=20,
		help='Steps between eval on test data')
	parser.add_argument('--tacotron_train_steps', type=int, default=1000000, help='total number of tacotron training steps')
	parser.add_argument('--wavenet_train_steps', type=int, default=500000, help='total number of wavenet training steps')
	parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
	parser.add_argument('--slack_url', default=None, help='slack webhook notification destination link')
	parser.add_argument('--emt_disc', action='store_true', default=False, help='whether to use emotion discriminator as part of loss')
	parser.add_argument('--spk_disc', action='store_true', default=False, help='whether to use speaker discriminator as part of loss')
	parser.add_argument('--intercross', action='store_true', default=True, help='whether to use intercross training')
	parser.add_argument('--synth_style_type', default=None, help='vary the emotion, speaker id, or neither')
	parser.add_argument('--tacotron_test_steps', type=int, default=3, help='Num batches to process when running evaluation')
	parser.add_argument('--remove_long_samps', action='store_true', default=False, help='Will remove out the longest samples from EMT4/VCTK')
	parser.add_argument('--test_max_len', action='store_true', default=False,help='Will create batches with the longest samples first to test max batch size')
	parser.add_argument('--unpaired', action='store_true', default=False,help='Will create batches with the longest samples first to test max batch size')
	parser.add_argument('--TEST', action='store_true', default=False,help='Uses small groups of batches to make testing faster')
	parser.add_argument('--TEST_INPUTS', action='store_true', default=False,help='Fixes all input data to be the same for testing')
	parser.add_argument('--max_to_keep', type=int, default=50, help='how many checkpoints to save')
	parser.add_argument('--recon_emb_loss', action='store_true', default=False, help='Adds loss for reconstructing embeddings')
	parser.add_argument('--intercross_both', action='store_true', default=False, help='does intercross for emotion and spk for both datasets')
	parser.add_argument('--intercross_spk_only', action='store_true', default=False,help='does intercross for emotion and spk for both datasets')
	parser.add_argument('--unpaired_loss_derate', type=float, default=1, help='how much to derate the unpaired mel out emb disc loss')
	parser.add_argument('--unpaired_emt_loss_derate', type=float, default=1,help='how much to derate the unpaired mel out emb disc loss')
	parser.add_argument('--lock_ref_enc', action='store_true', default=False, help='does not allow retraining of reference encoders')
	parser.add_argument('--lock_gst', action='store_true', default=False, help='does not allow retraining of global style tokens')
	parser.add_argument('--nat_gan', action='store_true', default=False, help='whether to use the naturalness gan')
	parser.add_argument('--restart_nat_gan_d', action='store_true', default=False, help='whether to use the naturalness gan')
	parser.add_argument('--nat_gan_derate', type=float, default=.1, help='how much to derate the unpaired mel out emb disc loss')
	parser.add_argument('--restore_nat_gan_d_sep', action='store_true', default=False,help='whether to use the naturalness gan')
	parser.add_argument('--save_output_vars', action='store_true', default=False, help='saves csvs of output vars')
	parser.add_argument('--opt_ref_no_mo', action='store_true', default=False, help='dont train encoders based on synthesized samples style embeddings')
	parser.add_argument('--restart_optimizer_r', action='store_true', default=False, help='retrains the reference encoder optimizer')
	parser.add_argument('--pretrained_emb_disc', action='store_true', default=False, help='whether to use pretrained emt disc')
	parser.add_argument('--pretrained_emb_disc_all', action='store_true', default=False,help='use pretrained emb disc on references and unpaired')
	parser.add_argument('--no_general', action='store_true', default=False, help='mel output loss is not being classified as general')
	parser.add_argument('--restore_std', action='store_true', default=False,help='allows the restoring of a model without optimzer_r to a new model with optimizer_r')
	parser.add_argument('--emt_attn', action='store_true', default=False,help='allows the restoring of a model without optimzer_r to a new model with optimizer_r')
	parser.add_argument('--emt_ref_gru', default='none', help='whether to use the a gru at the end of the reference embedding cnn')
	parser.add_argument('--emt_only', action='store_true', default=False,help='does only one condition - emotion')
	parser.add_argument('--attn', default=None, help='what type of attention to use')
	parser.add_argument('--up_ref_match_p', action='store_true', default=False, help='feeds in the same references as paired for unpaired')
	parser.add_argument('--tfr_up_only', action='store_true', default=False,help='feeds in the same references as paired for unpaired')
	parser.add_argument('--no_mo_style_loss', action='store_true', default=False,help='feeds in the same references as paired for unpaired')
	parser.add_argument('--l2_spk_emb', action='store_true', default=False,help='feeds in the same references as paired for unpaired')
	parser.add_argument('--flip_spk_emt', action='store_true', default=False,help='pass in emt as spk ref and vice versa - used for testing reversing the attention')
	parser.add_argument('--adain', action='store_true', default=False,help='use adaptive image normalization on references')
	parser.add_argument('--synth_constraint', action='store_true', default=False, help='use adaptive image normalization on references')
	parser.add_argument('--adv_emb_disc', action='store_true', default=False,help='use adversarial training on style embeddings')
	args = parser.parse_args()

	accepted_models = ['Tacotron', 'WaveNet', 'Tacotron-2']

	if args.model not in accepted_models:
		raise ValueError('please enter a valid model to train: {}'.format(accepted_models))

	log_dir, hparams = prepare_run(args)

	synth_metadata_filename = r"synth_emt4.txt"
	args.synth_metadata_filename = os.path.join(r"../data", synth_metadata_filename)
	import socket
	if socket.gethostname() in ['A3907623','MININT-39T168F']:
		hparams.tacotron_num_gpus = 1
		hparams.tacotron_batch_size = 32
	elif socket.gethostname() == 'area51.cs.washington.edu':
		hparams.tacotron_num_gpus = 2
		hparams.tacotron_batch_size = 64
		os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
		args.input_dir = '/data/tts_emotion'
		args.synth_metadata_filename = os.path.join(r"/data/tts_emotion",synth_metadata_filename)
		print("over-riding input directory for running on Area51")

	if hparams.tacotron_fine_tuning and not(args.restore):
		raise ValueError('fine_tuning set to true but not restoring the model!')

	#need to use the intercross both method with the zo/jessa datasets
	if args.tacotron_input.endswith('jessa.txt'):
		assert(args.intercross_both)

	if args.emt_attn and args.attn==None:
		raise ValueError("can't have emotion attention and no attention type")

	if args.flip_spk_emt:
		assert(not (args.unpaired))

	if args.model == 'Tacotron':
		tacotron_train(args, log_dir, hparams)
	elif args.model == 'WaveNet':
		wavenet_train(args, log_dir, hparams, args.wavenet_input)
	elif args.model == 'Tacotron-2':
		train(args, log_dir, hparams)
	else:
		raise ValueError('Model provided {} unknown! {}'.format(args.model, accepted_models))


if __name__ == '__main__':
	main()
