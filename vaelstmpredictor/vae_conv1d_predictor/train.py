"""
	Classifying Variational Autoencoder
"""

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical

from ..utils.model_utils import get_callbacks, save_model_in_pieces
from ..utils.model_utils import init_adam_wn, AnnealLossWeight
from ..utils.weightnorm import data_based_init

from .model import ConvVAEPredictor as VAEPredictor

def info_message(message, end='\n'): 
	print('[INFO] {}'.format(message), end=end)

def debug_message(message, end='\n'): 
	print('[DEBUG] {}'.format(message), end=end)

def warning_message(message, end='\n'): 
	print('[WARNING] {}'.format(message), end=end)

def train_vae_predictor(clargs, data_instance, network_type = 'Dense'):
	"""Training control operations to create VAEPredictor instance, 
		organize the input data, and train the network.
	
	Args:
		clargs (object): command line arguments from `argparse`
			Structure Contents: 
				clargs.n_labels
				clargs.predict_next
				clargs.use_prev_input
				clargs.run_name
				clargs.patience
				clargs.kl_anneal
				clargs.do_log
				clargs.do_ckpt
				clargs.num_epochs
				clargs.w_kl_anneal
				clargs.optimizer
				clargs.batch_size
		
		data_instance (object): object instance for organizing data structures
			Structure Contents: 
				DI.train_labels
				DI.valid_labels
				DI.test_labels
				DI.labels_train
				DI.data_train
				DI.labels_valid
				DI.data_valid

		test_test (optional; bool): flag for storing predictor test parameters
	
	Returns:
		vae_predictor (object): Variational AutoEncoder class instance
			Structure content: all data, all layers, training output, methods

		best_loss (dict): the best validation loss achieved during training
		
		history (object): output of the `keras` training procedures 
			Structure Contents:
				history (dict): loss, val_lss, etc
				epochs (list): list(range(num_epochs))
	"""
	config = tf.ConfigProto()
	# dynamically grow the memory used on the GPU
	config.gpu_options.allow_growth = True  

	# to log device placement (on which device the operation ran)
	# (nothing gets printed in Jupyter, only if you run it standalone)
	config.log_device_placement = False
	sess = tf.Session(config=config)

	# set this TensorFlow session as the default session for Keras
	set_session(sess)
	
	DI = data_instance

	clargs.n_labels = len(np.unique(DI.train_labels))

	predictor_train = DI.train_labels
	predictor_validation = DI.valid_labels
	
	callbacks = get_callbacks(clargs, patience=clargs.patience, 
					min_epoch = max(clargs.kl_anneal, clargs.w_kl_anneal)+1, 
					do_log = clargs.do_log, do_ckpt = clargs.do_ckpt)

	if clargs.kl_anneal > 0:
		assert(clargs.kl_anneal <= clargs.num_epochs), "invalid kl_anneal"
		vae_kl_weight = K.variable(value=0.1)
		callbacks += [AnnealLossWeight(vae_kl_weight, name="vae_kl_weight", 
								final_value=1.0, n_epochs=clargs.kl_anneal)]
	else:
		vae_kl_weight = 1.0
	if clargs.w_kl_anneal > 0:
		assert(clargs.w_kl_anneal <= clargs.num_epochs), "invalid w_kl_anneal"
		predictor_kl_weight = K.variable(value=0.0)
		callbacks += [AnnealLossWeight(predictor_kl_weight, name="predictor_kl_weight", 
								final_value=1.0, n_epochs=clargs.w_kl_anneal)]
	else:
		predictor_kl_weight = 1.0

	# clargs.optimizer, was_adam_wn = init_adam_wn(clargs.optimizer)

	if network_type.lower() == 'dense':
		info_message('Training Dense VAE Predictor'.format())
		vae_dims = (clargs.vae_hidden_dim, clargs.vae_latent_dim)
		predictor_dims = (clargs.predictor_hidden_dim, clargs.n_labels)
		vae_predictor = VAEPredictor(original_dim = clargs.original_dim, 
								vae_hidden_dims = [clargs.vae_hidden_dim], 
								dnn_hidden_dims =[clargs.predictor_hidden_dim],
								vae_latent_dim = clargs.vae_latent_dim, 
								dnn_out_dim = clargs.n_labels, 
								dnn_latent_dim = clargs.n_labels-1, 
								optimizer = 'adam')
	elif network_type.lower() == 'conv1d':
		info_message('Training Conv1D VAE Predictor'.format())
		data_shape = (784,1) # MNIST
		''' Configure dnn '''
		n_dnn_layers = clargs.num_dnn_layers
		
		dnn_filters = np.array([clargs.dnn_filter_size]*n_dnn_layers)
		dnn_filters = dnn_filters*(2**np.arange(n_dnn_layers))
		
		dnn_kernel_sizes = [clargs.dnn_kernel_size]*n_dnn_layers
		dnn_strides = [clargs.dnn_stride]*n_dnn_layers

		''' Configure encoder '''
		n_encoder_layers = clargs.num_encoder_layers
		
		encoder_filters = np.array([clargs.encoder_filter_size]*n_encoder_layers)
		encoder_filters = encoder_filters*(2**np.arange(n_encoder_layers))
		
		encoder_kernel_sizes = [clargs.encoder_kernel_size]*n_encoder_layers
		encoder_strides = [clargs.encoder_stride]*n_encoder_layers

		''' Configure Decoder '''
		n_decoder_layers = clargs.num_decoder_layers
		
		decoder_filters = np.array([clargs.decoder_filter_size]*n_decoder_layers)
		decoder_filters = decoder_filters//(2**np.arange(n_decoder_layers))

		decoder_kernel_sizes = [clargs.decoder_kernel_size]*n_decoder_layers
		decoder_strides = [clargs.decoder_stride]*n_decoder_layers

		vae_predictor = ConvVAEPredictor(
					encoder_filters = encoder_filters, 
					encoder_kernel_sizes = encoder_kernel_sizes,
					decoder_filters = decoder_filters, 
					decoder_kernel_sizes = decoder_kernel_sizes, 
					dnn_filters = dnn_filters, 
					dnn_kernel_sizes = dnn_kernel_sizes, 
					dnn_strides = dnn_strides,
					encoder_strides = encoder_strides, 
					decoder_strides = decoder_strides, 
					vae_latent_dim = clargs.vae_latent_dim, 
					encoder_top_size = clargs.encoder_top_size, 
					final_kernel_size = clargs.decoder_kernel_size,
					data_shape = data_shape, 
					batch_size = clargs.batch_size, 
					run_all = clargs.run_all, 
					verbose = clargs.verbose, 
					plot_model = clargs.plot_model,
					original_dim = clargs.original_dim, 
					dnn_out_dim = clargs.n_labels, 
					dnn_latent_dim = clargs.dnn_latent_dim, 
					dnn_log_var_prior = clargs.dnn_log_var_prior, 
					optimizer = clargs.optimizer,
					layer_type = clargs.network_type)
	else:
		ValueError('network_type must (currently) be either Dense or Conv1D')
	
	vae_predictor.build_model(dnn_weight = clargs.dnn_weight,
							vae_weight = clargs.vae_weight,
							dnn_kl_weight = clargs.dnn_kl_weight,
							vae_kl_weight = clargs.vae_kl_weight)
	
	# vae_predictor.compile(dnn_weight = clargs.dnn_weight,
	# 					vae_weight = clargs.vae_weight,
	# 					dnn_kl_weight = clargs.dnn_kl_weight,
	# 					vae_kl_weight = clargs.vae_kl_weight)

	# clargs.optimizer = 'adam-wn' if was_adam_wn else clargs.optimizer
	
	save_model_in_pieces(vae_predictor.model, clargs)
	
	# if clargs.use_prev_input:
	# 	vae_train = [DI.labels_train, DI.data_train]
	# 	vae_features_val = [DI.labels_valid, DI.data_valid]
	# else:
	data_based_init(vae_predictor.model, DI.data_train[:clargs.batch_size])

	if 'Conv1D'.lower() in clargs.network_type.lower():
		DI.data_train = np.expand_dims(DI.data_train, axis=2)
		DI.data_test = np.expand_dims(DI.data_test, axis=2)
		DI.data_valid = np.expand_dims(DI.data_valid, axis=2)
		DI.labels_train = np.expand_dims(DI.labels_train, axis=2)
		DI.labels_valid = np.expand_dims(DI.labels_valid, axis=2)
		predictor_train = np.expand_dims(predictor_train, axis=2)
		predictor_validation = np.expand_dims(predictor_validation, axis=2)
	
	if clargs.verbose:
		print()
		print('[INFO] DI.data_train.shape', DI.data_train.shape)
		print('[INFO] DI.data_test.shape', DI.data_test.shape)
		print('[INFO] DI.data_valid.shape', DI.data_valid.shape)
		print('[INFO] DI.labels_train.shape', DI.labels_train.shape)
		print()

	vae_train = DI.data_train
	vae_features_val = DI.data_valid
	
	vae_labels_val = [DI.labels_valid, predictor_validation, 
						predictor_validation,DI.labels_valid]
	validation_data = (vae_features_val, vae_labels_val)
	train_labels = [DI.labels_train, predictor_train, 
					predictor_train, DI.labels_train]
	
	if clargs.debug: return 0,0,0
	
	vae_predictor.model.summary()

	history = vae_predictor.model.fit(vae_train, train_labels,
								shuffle = True,
								epochs = clargs.num_epochs,
								batch_size = clargs.batch_size,
								callbacks = callbacks,
								validation_data = validation_data)

	best_ind = np.argmin([x if i >= max(clargs.kl_anneal,clargs.w_kl_anneal)+1\
				else np.inf for i,x in enumerate(history.history['val_loss'])])
	
	best_loss = {k: history.history[k][best_ind] for k in history.history}
	
	return vae_predictor, best_loss, history
