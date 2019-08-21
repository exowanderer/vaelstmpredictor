import argparse
import os

from glob import glob
from numpy import array, arange, vstack, reshape, loadtxt, zeros
from sklearn.externals import joblib
from time import time
from tqdm import tqdm

from vaelstmpredictor.vae_predictor.train import train_vae_predictor

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='deleteme',
                        help='tag for current run')
    parser.add_argument('--predictor_type', type=str, default="classification",
                        help='select `classification` or `regression`')
    parser.add_argument('--network_type', type=str, default='Conv1D',
                        help='Type of network to train: Dense or Conv1D')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer name')  # 'rmsprop'
    parser.add_argument('--num_epochs', type=int, default=2,
                        help='number of epochs')
    parser.add_argument('--original_dim', type=int, default=0,
                        help='input dim')
    parser.add_argument('--run_all', action='store_true',
                        help='Boolean for starting training now!')
    parser.add_argument('--vae_latent_dim', type=int, default=2,
                        help='Size of the latent layer')
    parser.add_argument('--dnn_filter_size', type=int, default=16,
                        help='Number of filters in the dnn')
    parser.add_argument('--dnn_kernel_size', type=int, default=3,
                        help='Kernel size across the dnn')
    parser.add_argument('--dnn_stride', type=int, default=2,
                        help='Stride size across the dnn')
    parser.add_argument('--num_dnn_layers', type=int, default=2,
                        help='Number of layers in the dnn')
    parser.add_argument('--dnn_top_size', type=int, default=16,
                        help='Size of Dense layer on top of dnn')
    parser.add_argument('--encoder_filter_size', type=int, default=16,
                        help='Number of filters in the encoder')
    parser.add_argument('--encoder_kernel_size', type=int, default=3,
                        help='Kernel size across the encoder')
    parser.add_argument('--encoder_stride', type=int, default=2,
                        help='Stride size across the encoder')
    parser.add_argument('--num_encoder_layers', type=int, default=2,
                        help='Number of layers in the encoder')
    parser.add_argument('--encoder_top_size', type=int, default=16,
                        help='Size of Dense layer on top of encoder')
    parser.add_argument('--decoder_filter_size', type=int, default=32,
                        help='Number of filters in the decoder')
    parser.add_argument('--decoder_kernel_size', type=int, default=3,
                        help='Kernel size across the decoder')
    parser.add_argument('--decoder_stride', type=int, default=2,
                        help='Stride size across the decoder')
    parser.add_argument('--num_decoder_layers', type=int, default=2,
                        help='Number of layers in the decoder')
    parser.add_argument('--decoder_bottom_size', type=int, default=16,
                        help='Size of Dense layer on top of decoder')
    parser.add_argument('--seq_length', type=int, default=1,
                        help='sequence length (concat)')
    parser.add_argument('--dnn_weight', type=float, default=1.0,  # 5716223
                        help='relative weight on classifying key')
    parser.add_argument('--vae_weight', type=float, default=1.0,  # 1.002,
                        help='relative weight on classifying key')
    parser.add_argument('--vae_kl_weight', type=float, default=1.0,  # 1.0,
                        help='relative weight on classifying key')
    parser.add_argument('--dnn_kl_weight', type=float, default=1.0,  # 476,
                        help='relative weight on classifying key')
    parser.add_argument('--dnn_latent_dim', type=int, default=9,
                        help='predictor dims for class/regr prediction')
    parser.add_argument('--dnn_log_var_prior', type=float, default=0.0,
                        help='Prion the Log Variances over the DNN')
    parser.add_argument("--do_log", action="store_true",
                        help="save log files")
    parser.add_argument("--do_ckpt", action="store_true",
                        help="save model checkpoints")
    parser.add_argument('--patience', type=int, default=10,
                        help='# of epochs, for early stopping')
    parser.add_argument('--log_dir', type=str, default='../data/logs',
                        help='basedir for saving log files')
    parser.add_argument('--model_dir', type=str, default='../data/models',
                        help='basedir for saving model weights')
    parser.add_argument('--train_file', type=str,
                        default='../data/input/JSB Chorales_Cs.pickle',
                        help='file of training data (.pickle)')
    parser.add_argument('--no_squeeze_x', action="store_true",
                        help='whether to squeeze the x dimension')
    parser.add_argument('--no_squeeze_y', action="store_true",
                        help='whether to squeeze the x dimension')
    parser.add_argument('--step_length', type=int, default=1,
                        help="Length of the step for overlap in song(s)")
    parser.add_argument('--debug', action="store_true",
                        help="if debug; then stop before model.fit")
    parser.add_argument('--plot_model', action="store_true",
                        help='Toggle whether to plot the model using keras.utils.plot_model')
    parser.add_argument('--plot_name', type=str,
                        default='Conv1D_VAE_Model_Diagram.png',
                        help='Image file name to save model diagram')
    parser.add_argument('--verbose', action='store_true',
                        help='Toggle whether to print more statements.')
    parser.add_argument("--kl_anneal", type=int, default=0,
                        help="number of epochs before kl loss term is 1.0")
    parser.add_argument("--w_kl_anneal", type=int, default=0,
                        help="number of epochs before w's kl loss term is 1.0")

    clargs = parser.parse_args()

    """
	'''Weights Determined from First Few Run-throughs
		All weights are relative to sum-normalization
	'''
	vae_weight = 36.34479883984246 # ~ 36.4
	dnn_weight = 1.190315303141271 # ~ 1.2
	vae_kl_weight = 1657184.0548487047 # ~ 1.7e6
	dnn_kl_weight = 7.55449024567151 # ~ 7.6

	''' All weights normalized to largest loss'''
	vae_weight 30.5337575211607 # ~30.53
	dnn_weight = 1.0 # ~1.00
	dnn_kl_weight =  6.346629523904319 # ~6.35
	vae_kl_weight 1392222.7585206672 # ~1.39e6
	"""

    if 'class' in clargs.predictor_type.lower():
        clargs.predictor_type = 'classification'
    if 'regr' in clargs.predictor_type.lower():
        clargs.predictor_type = 'regression'

    if clargs.predictor_type is 'regression':
        clargs.n_labels = 1

    train_files = ['piano', 'mnist', 'exoplanet']

    if 'piano' in clargs.train_file.lower():
        from vaelstmpredictor.utils.data_utils import PianoData

        clargs.train_file = 'PianoData'

        clargs.predict_next = False
        clargs.use_prev_input = False
        return_label_next = clargs.predict_next or clargs.use_prev_input

        P = PianoData(train_file=clargs.train_file,
                      batch_size=clargs.batch_size,
                      seq_length=clargs.seq_length,
                      step_length=clargs.step_length,
                      return_label_next=return_label_next,
                      squeeze_x=not clargs.no_squeeze_x,
                      squeeze_y=not clargs.no_squeeze_y)

        # Keep default unless modified inside `PianoData` instance
        clargs.original_dim = P.original_dim or clargs.original_dim

        data_instance = P

    elif 'mnist' in clargs.train_file.lower():
        from vaelstmpredictor.utils.data_utils import MNISTData

        clargs.train_file = 'mnist'
        data_instance = MNISTData(batch_size=clargs.batch_size)

    elif 'exoplanet' in clargs.train_file.lower():
        from vaelstmpredictor.utils.data_utils import ExoplanetData

        clargs.train_file = 'exoplanet'
        data_instance = ExoplanetData(train_file=None,  # clargs.train_file,
                                      batch_size=clargs.batch_size)
    else:
        raise ValueError("`train_file` must be in list {}".format(train_files))

    n_train, n_features = data_instance.data_train.shape
    n_test, n_features = data_instance.data_valid.shape

    if clargs.original_dim is 0:
        clargs.original_dim = n_features

    time_stmp = int(time())
    clargs.run_name = '{}_{}_{}'.format(
        clargs.run_name, clargs.train_file, time_stmp)

    print('\n\n[INFO] Run Base Name: {}\n'.format(clargs.run_name))

    vae_model, best_loss, history = train_vae_predictor(clargs=clargs,
                                                        data_instance=data_instance,
                                                        network_type=clargs.network_type)

    print('\n\n[INFO] The Best Loss: {}\n'.format(best_loss))
    joblib_save_loc = '{}/{}_{}_trained_model_output.joblib.save'.format(
        clargs.model_dir,
        clargs.run_name,
        clargs.network_type)

    weights_save_loc = '{}/{}_{}_trained_model_weights.save'.format(
        clargs.model_dir,
        clargs.run_name,
        clargs.network_type)

    model_save_loc = '{}/{}_{}_trained_model_full.save'.format(
        clargs.model_dir,
        clargs.run_name,
        clargs.network_type)

    vae_model.model.save_weights(weights_save_loc, overwrite=True)
    vae_model.model.save(model_save_loc, overwrite=True)

    joblib.dump({'best_loss': best_loss, 'history': history}, joblib_save_loc)
